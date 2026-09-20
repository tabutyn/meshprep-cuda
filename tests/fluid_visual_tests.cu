// SPDX-License-Identifier: MIT
#include "../apps/water_lab/fluid_visuals.hpp"
#include "../apps/water_lab/fluid_surface.cuh"
#include "../apps/water_lab/particle_cells.cuh"

#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status, const char* message)
{
    if (status != cudaSuccess) throw std::runtime_error(message);
}

template <typename T> class DeviceBuffer {
public:
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    void upload(const std::vector<T>& values)
    {
        if (!data_) check(cudaMalloc(reinterpret_cast<void**>(&data_),values.size()*sizeof(T)),
            "allocate fixture buffer");
        check(cudaMemcpy(data_,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice),
            "upload fixture buffer");
    }
    std::vector<T> download(std::size_t count) const
    {
        std::vector<T> values(count);
        check(cudaMemcpy(values.data(),data_,count*sizeof(T),cudaMemcpyDeviceToHost),
            "download fixture buffer");
        return values;
    }
    T* get() const { return data_; }
private:
    T* data_{};
};

bool bytes_equal(const auto& a, const auto& b)
{
    return a.size()==b.size() &&
        std::memcmp(a.data(),b.data(),a.size()*sizeof(typename std::decay_t<decltype(a)>::value_type))==0;
}
void validate(const std::vector<float4>& values)
{
    for (float4 value : values) {
        require(std::isfinite(value.x)&&std::isfinite(value.y)&&
                std::isfinite(value.z)&&std::isfinite(value.w),"non-finite visual output");
        const float normal=std::sqrt(value.x*value.x+value.y*value.y+value.z*value.z);
        require(normal<=1.0e-5F || std::abs(normal-1.0F)<=2.0e-3F,
            "distribution normal is neither unit length nor zero");
        require(value.w>=0.0F && value.w<=1.0F,"foam coverage is outside [0,1]");
    }
}
std::size_t active_count(const std::vector<waterlab::FoamParticle>& values)
{
    return std::count_if(values.begin(),values.end(),[](const auto& p) { return p.active(); });
}
void validate_foam(const std::vector<waterlab::FoamParticle>& values)
{
    require(values.size()==waterlab::FluidVisuals::foam_capacity,"foam capacity changed");
    for (const auto& p : values) {
        for (const auto row : {p.position_age,p.velocity_life,p.normal_radius}) {
            require(std::isfinite(row.x)&&std::isfinite(row.y)&&std::isfinite(row.z)&&
                std::isfinite(row.w),"non-finite secondary foam state");
        }
        if (p.active()) {
            require(p.position_age.w<p.velocity_life.w,"expired foam remains active");
            require(p.normal_radius.w>0.0F && p.normal_radius.w<0.05F,"invalid foam render radius");
        }
    }
}

struct Fixture {
    static constexpr float radius=0.0225F, spacing=0.06F, support=0.11F;
    std::vector<float3> positions, zero, uniform, energetic;
    std::vector<meshprep::Aabb> bounds;
    DeviceBuffer<float3> device_positions, device_velocities;
    DeviceBuffer<meshprep::Aabb> device_bounds;
    DeviceBuffer<std::uint64_t> device_cell_keys;
    DeviceBuffer<std::uint32_t> device_cell_indices;
    meshprep::Workspace workspace;
    meshprep::Hierarchy hierarchy;

    Fixture()
    {
        for (int y=-3; y<=3; ++y) for (int z=-3; z<=3; ++z) for (int x=-3; x<=3; ++x) {
            const float3 p=make_float3(x*spacing,y*spacing,z*spacing);
            positions.push_back(p);
            zero.push_back({});
            uniform.push_back(make_float3(0.8F,0.4F,-0.3F));
            energetic.push_back(y==3
                ? make_float3(((x+z)&1)?2.5F:-2.5F,2.5F,0.0F) : float3{});
            bounds.push_back({make_float3(p.x-radius,p.y-radius,p.z-radius),
                make_float3(p.x+radius,p.y+radius,p.z+radius)});
        }
        device_positions.upload(positions); device_velocities.upload(zero); device_bounds.upload(bounds);
        std::vector<std::pair<std::uint64_t,std::uint32_t>> cells;
        cells.reserve(positions.size());
        for (std::uint32_t i=0U; i<positions.size(); ++i) {
            const float3 p=positions[i];
            cells.emplace_back(waterlab::detail::particle_cell_key(
                static_cast<int>(std::floor(p.x/support)),
                static_cast<int>(std::floor(p.y/support)),
                static_cast<int>(std::floor(p.z/support))), i);
        }
        std::stable_sort(cells.begin(),cells.end(),[](const auto& left,const auto& right) {
            return left.first<right.first;
        });
        std::vector<std::uint64_t> keys;
        std::vector<std::uint32_t> indices;
        keys.reserve(cells.size()); indices.reserve(cells.size());
        for (const auto [key,index] : cells) { keys.push_back(key); indices.push_back(index); }
        device_cell_keys.upload(keys); device_cell_indices.upload(indices);
        const auto status=meshprep::build_hierarchy(
            {device_bounds.get(),bounds.size()},{},workspace,hierarchy);
        require(status.ok(),"failed to build visual fixture hierarchy");
    }
    [[nodiscard]] waterlab::ParticleCellView cells() const noexcept
    {
        return {device_positions.get(),device_cell_keys.get(),device_cell_indices.get(),
            static_cast<std::uint32_t>(positions.size()),support};
    }
    void require_inputs(const std::vector<float3>& velocities) const
    {
        require(bytes_equal(device_positions.download(positions.size()),positions),
            "visual update changed particle positions");
        require(bytes_equal(device_velocities.download(velocities.size()),velocities),
            "visual update changed particle velocities");
    }
};

std::vector<float4> update(waterlab::FluidVisuals& visuals, Fixture& fixture,
    const std::vector<float3>& velocities, float dt, unsigned count=1U,
    waterlab::ParticleCellView cells={})
{
    fixture.device_velocities.upload(velocities);
    for (unsigned i=0; i<count; ++i) {
        const float milliseconds=visuals.update(fixture.device_positions.get(),
            fixture.device_velocities.get(),fixture.hierarchy,Fixture::support,
            make_float3(0,-9.81F,0),dt,nullptr,false,cells);
        require(std::isfinite(milliseconds)&&milliseconds>=0.0F,"invalid visual timing");
    }
    fixture.require_inputs(velocities);
    std::vector<float4> output; visuals.capture(output); validate(output); return output;
}

void test_cell_queries(Fixture& fixture)
{
    waterlab::FluidSurface bvh_surface(32U), cell_surface(32U);
    const auto particle_count=static_cast<std::uint32_t>(fixture.positions.size());
    (void)bvh_surface.update(fixture.device_positions.get(),particle_count,
        fixture.hierarchy,Fixture::support);
    (void)cell_surface.update(fixture.device_positions.get(),particle_count,
        fixture.hierarchy,Fixture::support,nullptr,fixture.cells());
    const std::size_t sample_count=32U*32U*32U;
    std::vector<float> bvh(sample_count),cells(sample_count);
    check(cudaMemcpy(bvh.data(),bvh_surface.view().values,sample_count*sizeof(float),
        cudaMemcpyDeviceToHost),"download BVH field");
    check(cudaMemcpy(cells.data(),cell_surface.view().values,sample_count*sizeof(float),
        cudaMemcpyDeviceToHost),"download cell field");
    for (std::size_t i=0U; i<sample_count; ++i) {
        require(std::abs(bvh[i]-cells[i])<2.0e-5F,
            "cell surface differs from exact BVH surface");
    }

    waterlab::FluidVisuals bvh_visuals(particle_count,32U);
    waterlab::FluidVisuals cell_visuals(particle_count,32U);
    const auto bvh_output=update(bvh_visuals,fixture,fixture.energetic,1.0F/60.0F);
    cell_visuals.set_active_count(256U);
    const auto cell_output=update(
        cell_visuals,fixture,fixture.energetic,1.0F/60.0F,1U,fixture.cells());
    require(cell_visuals.view().particle_count==particle_count,
        "particle cell view did not synchronize resized fluid visuals");
    for (std::size_t i=0U; i<bvh_output.size(); ++i) {
        require(std::abs(bvh_output[i].x-cell_output[i].x)<2.0e-5F &&
            std::abs(bvh_output[i].y-cell_output[i].y)<2.0e-5F &&
            std::abs(bvh_output[i].z-cell_output[i].z)<2.0e-5F &&
            std::abs(bvh_output[i].w-cell_output[i].w)<2.0e-5F,
            "cell distribution query differs from exact BVH query");
    }

    (void)update(bvh_visuals,fixture,fixture.energetic,1.0F/60.0F,7U);
    std::vector<waterlab::FoamParticle> generated_foam;
    std::uint64_t generated_tick{};
    bvh_visuals.capture_foam(generated_foam,generated_tick);
    const auto generated=std::find_if(generated_foam.begin(),generated_foam.end(),
        [](const auto& particle) { return particle.active(); });
    require(generated!=generated_foam.end(),"cell velocity oracle could not seed live foam");
    std::vector<waterlab::FoamParticle> seeded(waterlab::FluidVisuals::foam_capacity);
    seeded[0]=*generated;
    seeded[0].position_age.w=0.25F;
    seeded[0].velocity_life.w=2.0F;
    bvh_visuals.restore_foam(seeded,generated_tick);
    cell_visuals.restore_foam(seeded,generated_tick);
    (void)update(bvh_visuals,fixture,fixture.uniform,1.0F/60.0F);
    (void)update(cell_visuals,fixture,fixture.uniform,1.0F/60.0F,1U,fixture.cells());
    std::vector<waterlab::FoamParticle> bvh_foam,cell_foam;
    std::uint64_t bvh_tick{},cell_tick{};
    bvh_visuals.capture_foam(bvh_foam,bvh_tick);
    cell_visuals.capture_foam(cell_foam,cell_tick);
    require(bvh_tick==cell_tick && bvh_foam[0].active() && cell_foam[0].active(),
        "cell velocity query changed foam lifetime");
    const auto close4=[](float4 left,float4 right) {
        return std::abs(left.x-right.x)<2.0e-5F &&
            std::abs(left.y-right.y)<2.0e-5F &&
            std::abs(left.z-right.z)<2.0e-5F &&
            std::abs(left.w-right.w)<2.0e-5F;
    };
    require(close4(bvh_foam[0].position_age,cell_foam[0].position_age) &&
        close4(bvh_foam[0].velocity_life,cell_foam[0].velocity_life) &&
        close4(bvh_foam[0].normal_radius,cell_foam[0].normal_radius),
        "cell velocity query differs from exact BVH foam advection");

    bool rejected=false;
    auto invalid=fixture.cells();
    invalid.indexed_positions=fixture.device_velocities.get();
    try { (void)cell_visuals.update(fixture.device_positions.get(),fixture.device_velocities.get(),
        fixture.hierarchy,Fixture::support,make_float3(0,-9.81F,0),0.0F,nullptr,false,invalid); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"cell view for a different position buffer was accepted");
    rejected=false;
    invalid=fixture.cells(); invalid.cell_size=Fixture::support*2.0F;
    try { (void)cell_surface.update(fixture.device_positions.get(),particle_count,
        fixture.hierarchy,Fixture::support,nullptr,invalid); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"cell view with the wrong width was accepted");
}

void test_sampling()
{
    waterlab::FluidSurfaceGrid grid{make_float3(-0.3F,-0.2F,-0.1F),
        make_float3(0.2F,0.3F,0.4F),make_uint3(3,4,2),0.12F};
    const auto polynomial=[](float3 p) {
        return 1+2*p.x-3*p.y+0.5F*p.z+0.2F*p.x*p.y-0.4F*p.x*p.z+
            0.7F*p.y*p.z+0.1F*p.x*p.y*p.z;
    };
    std::vector<float> values;
    for (unsigned z=0; z<2; ++z) for (unsigned y=0; y<4; ++y) for (unsigned x=0; x<3; ++x) {
        values.push_back(polynomial(make_float3(grid.minimum.x+x*grid.cell_size.x,
            grid.minimum.y+y*grid.cell_size.y,grid.minimum.z+z*grid.cell_size.z)));
    }
    const waterlab::FluidSurfaceView view{values.data(),&grid};
    for (unsigned i=0; i<19; ++i) {
        const float t=static_cast<float>(i)/18.0F;
        const float3 p=make_float3(-0.3F+0.4F*t,-0.2F+0.9F*(1-t),-0.1F+0.4F*t);
        require(std::abs(waterlab::surface_sample(view,p)-polynomial(p))<2.0e-5F,
            "trilinear field differs from analytic polynomial");
        const float3 gradient=waterlab::surface_gradient(view,p);
        require(std::abs(gradient.x-(2+0.2F*p.y-0.4F*p.z+0.1F*p.y*p.z))<2.0e-5F &&
            std::abs(gradient.y-(-3+0.2F*p.x+0.7F*p.z+0.1F*p.x*p.z))<2.0e-5F &&
            std::abs(gradient.z-(0.5F-0.4F*p.x+0.7F*p.y+0.1F*p.x*p.y))<2.0e-5F,
            "trilinear gradient differs from analytic derivative");
    }
    require(waterlab::surface_sample(view,make_float3(2,2,2))>0,"outside grid is not positive");
    const float3 boundary=make_float3(grid.minimum.x+2*grid.cell_size.x,0,0);
    const float3 rounded=make_float3(std::nextafter(boundary.x,1.0F),0,0);
    require(std::abs(waterlab::surface_sample(view,boundary)-waterlab::surface_sample(view,rounded))<1.0e-5F,
        "one-ulp grid-boundary roundoff creates a false air jump");
}

void test_cubic_crossings()
{
    const auto check_root=[](auto polynomial, float expected, float scale=1.0F) {
        float root=-1;
        require(waterlab::surface_cubic_first_root(scale*polynomial(0.0F),
            scale*polynomial(1.0F/3.0F),scale*polynomial(2.0F/3.0F),scale*polynomial(1.0F),root),
            "cubic surface crossing was missed");
        require(std::abs(root-expected)<1.0e-5F,"cubic solver did not return the first crossing");
    };
    const auto double_cross=[](float u) { return (u-0.2F)*(u-0.8F)*(u+1); };
    check_root(double_cross,0.2F); // Both endpoints positive, interior negative.
    check_root(double_cross,0.2F,1.0e-12F); // Classification must not depend on field scale.
    check_root([](float u) { return (u-0.4F)*(u-0.4F)*(u+1); },0.4F);
    check_root([](float u) { return (u-0.1F)*(u-0.4F)*(u-0.9F); },0.1F);
    check_root([](float u) { return 1-2*u; },0.5F); // Enter water.
    check_root([](float u) { return 2*u-1; },0.5F); // Exit water.
    check_root([](float u) { return u; },0.0F);
    check_root([](float u) { return u-1; },1.0F);
    check_root([](float u) { return u-1.0e-5F; },1.0e-5F);
    check_root([](float u) { return u-0.99999F; },0.99999F);
    float root{};
    const auto positive=[](float u) { return (u-0.4F)*(u-0.4F)+0.02F; };
    require(!waterlab::surface_cubic_first_root(positive(0),positive(1.0F/3),
        positive(2.0F/3),positive(1),root),"positive cubic segment produced a false crossing");
    require(!waterlab::surface_cubic_first_root(0,1,std::numeric_limits<float>::quiet_NaN(),1,root),
        "non-finite cubic samples were accepted");

    // Restrict an actual trilinear cell to forward/reversed/cropped diagonal rays.
    waterlab::FluidSurfaceGrid grid{make_float3(0,0,0),make_float3(1,1,1),make_uint3(2,2,2),0.12F};
    float values[8]{};
    for (unsigned z=0; z<2; ++z) for (unsigned y=0; y<2; ++y) for (unsigned x=0; x<2; ++x)
        values[(z*2+y)*2+x]=(x-0.2F)*(y-0.8F)*(z+1);
    const waterlab::FluidSurfaceView view{values,&grid};
    for (const auto endpoints : {make_float3(0,1,0.2F),make_float3(1,0,0.2F),
             make_float3(0.1F,0.9F,0.125F),make_float3(0.9F,0.1F,0.125F)}) {
        check_root([&](float u) {
            const float t=endpoints.x+(endpoints.y-endpoints.x)*u;
            return waterlab::surface_sample(view,make_float3(t,t,t));
        },endpoints.z);
    }
}

void test_particle_surface()
{
    constexpr float support=0.12F;
    // Radius h/3=.04: individual spheres separated by .09 leave a .01 gap.
    const std::vector<float3> positions{{-0.045F,0,0},{0.045F,0,0}};
    const std::vector<meshprep::Aabb> bounds{{positions[0],positions[0]},{positions[1],positions[1]}};
    DeviceBuffer<float3> device_positions; device_positions.upload(positions);
    DeviceBuffer<meshprep::Aabb> device_bounds; device_bounds.upload(bounds);
    meshprep::Workspace workspace;
    meshprep::Hierarchy hierarchy;
    require(meshprep::build_hierarchy({device_bounds.get(),bounds.size()},{},workspace,hierarchy).ok(),
        "build surface fixture hierarchy");
    waterlab::FluidSurface surface(32U);
    require(surface.update(device_positions.get(),2U,hierarchy,support)>=0,"invalid surface timing");
    waterlab::FluidSurfaceGrid grid{};
    check(cudaMemcpy(&grid,surface.view().grid,sizeof(grid),cudaMemcpyDeviceToHost),"download surface grid");
    require(grid.dimensions.x > grid.dimensions.y &&
            grid.dimensions.x > grid.dimensions.z,
        "elongated fluid bounds did not receive an aspect-aware surface grid");
    std::vector<float> values(grid.dimensions.x*grid.dimensions.y*grid.dimensions.z);
    check(cudaMemcpy(values.data(),surface.view().values,values.size()*sizeof(float),
        cudaMemcpyDeviceToHost),"download surface field");
    for (std::size_t i=0; i<values.size(); ++i) {
        require(std::isfinite(values[i]),"non-finite scalar surface field");
        if (i%41U!=0U) continue;
        const std::size_t x=i%grid.dimensions.x;
        const std::size_t y=(i/grid.dimensions.x)%grid.dimensions.y;
        const std::size_t z=i/(grid.dimensions.x*grid.dimensions.y);
        const float3 p=make_float3(grid.minimum.x+x*grid.cell_size.x,
            grid.minimum.y+y*grid.cell_size.y,
            grid.minimum.z+z*grid.cell_size.z);
        double sum=0, dx=0, dy=0, dz=0;
        for (const auto q : positions) {
            const double x=static_cast<double>(p.x)-q.x, y=static_cast<double>(p.y)-q.y;
            const double z=static_cast<double>(p.z)-q.z;
            const double a=std::max(0.0,1-(x*x+y*y+z*z)/(static_cast<double>(support)*support));
            const double w=a*a*a;
            sum+=w; dx+=w*x; dy+=w*y; dz+=w*z;
        }
        const double expected=sum>1.0e-10 ? std::sqrt(dx*dx+dy*dy+dz*dz)/sum-support/3.0 : support;
        require(std::abs(values[i]-expected)<2.0e-5,"GPU field differs from weighted-center CPU reference");
    }
    const waterlab::FluidSurfaceView host{values.data(),&grid};
    for (int i=-10; i<=10; ++i) {
        require(waterlab::surface_sample(host,make_float3(i*0.0045F,0,0))<0,
            "nearby particles have no continuous liquid bridge");
    }
    require(waterlab::surface_sample(host,make_float3(0,0.10F,0))>0,"fluid field swallows exterior air");
    bool rejected=false;
    try { (void)surface.update(device_positions.get(),3U,hierarchy,support); }
    catch (const std::runtime_error&) { rejected=true; }
    require(rejected,"hierarchy/particle count mismatch was accepted");
    rejected=false;
    try { (void)surface.update(device_positions.get(),0U,hierarchy,support); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"empty particle surface was accepted");
}

void run_tests()
{
    test_sampling();
    test_cubic_crossings();
    test_particle_surface();
    Fixture fixture;
    test_cell_queries(fixture);
    waterlab::FluidVisuals visuals(static_cast<std::uint32_t>(fixture.positions.size()),32U);
    require(visuals.view().particle_count==fixture.positions.size(),"visual view count is wrong");
    require(visuals.allocated_bytes()>=fixture.positions.size()*sizeof(float4),
        "visual allocation accounting is too small");
    visuals.set_foam_settings({12.0F, 2.0F, 1.5F});
    const auto settings = visuals.foam_settings();
    require(settings.emission_rate == 12.0F && settings.radius_scale == 2.0F &&
            settings.lifetime_scale == 1.5F,
        "foam settings were not retained");
    bool invalid_settings_rejected = false;
    try {
        visuals.set_foam_settings({65.0F, 2.0F, 1.5F});
    } catch (const std::invalid_argument&) {
        invalid_settings_rejected = true;
    }
    require(invalid_settings_rejected, "out-of-range foam settings were accepted");
    visuals.set_foam_settings({});

    visuals.reset();
    (void)update(visuals,fixture,fixture.zero,1.0F/60.0F);
    std::vector<waterlab::FoamParticle> foam;
    std::uint64_t tick{};
    visuals.capture_foam(foam,tick);
    require(active_count(foam)==0U,"stationary particles generated foam");
    visuals.reset();
    (void)update(visuals,fixture,fixture.uniform,1.0F/60.0F);
    visuals.capture_foam(foam,tick);
    require(active_count(foam)==0U,"uniform translation generated foam");

    const auto energetic_run=[&] {
        visuals.reset(); return update(visuals,fixture,fixture.energetic,1.0F/60.0F,8U);
    };
    const auto first=energetic_run();
    std::vector<waterlab::FoamParticle> first_foam;
    std::uint64_t first_tick{};
    visuals.capture_foam(first_foam,first_tick); validate_foam(first_foam);
    require(active_count(first_foam)>0U,"energetic upward free surface generated no secondary foam");
    require(std::any_of(first_foam.begin(),first_foam.end(),[](const auto& p) {
        return p.active() && p.position_age.y>0.12F;
    }),"foam birth missed the upward free surface");
    waterlab::FluidSurfaceGrid attached_grid{};
    const auto device_surface=visuals.view().surface;
    check(cudaMemcpy(&attached_grid,device_surface.grid,sizeof(attached_grid),
        cudaMemcpyDeviceToHost),"download foam attachment grid");
    std::vector<float> attached_values(attached_grid.dimensions.x*attached_grid.dimensions.y*
        attached_grid.dimensions.z);
    check(cudaMemcpy(attached_values.data(),device_surface.values,attached_values.size()*sizeof(float),
        cudaMemcpyDeviceToHost),"download foam attachment field");
    const waterlab::FluidSurfaceView attached_surface{attached_values.data(),&attached_grid};
    for (const auto& p : first_foam) if (p.active()) {
        const float3 position=make_float3(p.position_age.x,p.position_age.y,p.position_age.z);
        const float value=waterlab::surface_sample(attached_surface,position);
        const float3 gradient=waterlab::surface_gradient(attached_surface,position);
        const float length=std::sqrt(gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z);
        require(length>1.0e-6F && std::abs(value)/length<=
            0.3F*p.normal_radius.w+0.02F*Fixture::support+1.0e-5F,
            "active foam is detached from the reconstructed liquid surface");
    }
    std::printf("INFO energetic fixture: %zu active foam particles, attachment residuals passed\n",
        active_count(first_foam));
    std::vector<float4> before_pause; visuals.capture(before_pause);
    const auto paused=update(visuals,fixture,fixture.energetic,0.0F);
    require(bytes_equal(before_pause,paused),"zero-dt update changed visual state");
    visuals.capture_foam(foam,tick);
    require(bytes_equal(first_foam,foam)&&first_tick==tick,"zero-dt update aged/moved/spawned foam");
    (void)update(visuals,fixture,fixture.uniform,1.0F/60.0F);
    visuals.capture_foam(foam,tick); validate_foam(foam);
    bool advected=false;
    for (std::size_t i=0; i<foam.size(); ++i) {
        if (foam[i].active() && first_foam[i].active()) {
            advected=advected || std::abs(foam[i].position_age.x-first_foam[i].position_age.x)>1.0e-5F;
        }
    }
    require(advected,"secondary foam did not advect with water velocity");
    float expiry=0;
    for (const auto& p : first_foam) expiry=std::max(expiry,p.velocity_life.w);
    (void)update(visuals,fixture,fixture.zero,expiry+0.1F);
    visuals.capture_foam(foam,tick); validate_foam(foam);
    require(active_count(foam)==0U,"secondary foam did not expire after finite lifetime");
    visuals.restore(first);
    visuals.restore_foam(first_foam,first_tick);
    std::vector<float4> restored; visuals.capture(restored);
    require(bytes_equal(first,restored),"visual snapshot restore is not exact");
    visuals.capture_foam(foam,tick);
    require(bytes_equal(first_foam,foam)&&tick==first_tick,"foam snapshot restore is not exact");
    auto zero_inactive=first_foam;
    for (auto& p : zero_inactive) if (!p.active()) p={};
    zero_inactive.back()={}; // Ensure the all-zero inactive representation is exercised.
    visuals.restore_foam(zero_inactive,first_tick);
    (void)update(visuals,fixture,fixture.energetic,0.0F);
    visuals.capture_foam(foam,tick);
    require(bytes_equal(zero_inactive,foam)&&tick==first_tick,
        "paused update rewrote restored zero-inactive foam records");
    const auto second=energetic_run();
    require(bytes_equal(first,second),"reset visual update is not deterministic");
    visuals.capture_foam(foam,tick);
    require(bytes_equal(first_foam,foam)&&tick==first_tick,"reset secondary foam is not deterministic");

    bool rejected=false;
    try { std::vector<float4> invalid(first.size()-1U); visuals.restore(invalid); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid visual snapshot size was accepted");
    rejected=false;
    try { auto invalid=first_foam; invalid[0].position_age.x=std::numeric_limits<float>::quiet_NaN();
        visuals.restore_foam(invalid,first_tick); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"non-finite foam snapshot was accepted");
    rejected=false;
    try { (void)visuals.update(fixture.device_positions.get(),fixture.device_velocities.get(),
        fixture.hierarchy,Fixture::support,make_float3(0,-9.81F,0),
        std::numeric_limits<float>::quiet_NaN()); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"NaN visual timestep was accepted");
    fixture.require_inputs(fixture.energetic);
}

} // namespace

int main()
{
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) {
        std::puts("SKIP meshprep-fluid-visual-tests: no CUDA device"); return 77;
    }
    try { run_tests(); std::puts("PASS meshprep-fluid-visual-tests"); return 0; }
    catch (const std::exception& error) {
        std::fprintf(stderr,"FAIL meshprep-fluid-visual-tests: %s\n",error.what()); return 1;
    }
}
