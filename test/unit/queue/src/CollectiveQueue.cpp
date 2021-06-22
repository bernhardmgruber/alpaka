/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/alpaka.hpp>
#    include <alpaka/test/queue/Queue.hpp>
#    include <alpaka/test/queue/QueueCpuOmp2Collective.hpp>
#    include <alpaka/test/queue/QueueTestFixture.hpp>

#    include <catch2/catch.hpp>

#    include <vector>

struct QueueCollectiveTestKernel
{
    template<typename TAcc>
    auto operator()(TAcc const& acc, int* results_ptr) const -> void
    {
        size_t thread_id = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
        // avoid that one thread is doing all the work
        std::this_thread::sleep_for(std::chrono::milliseconds(200u * thread_id));
        results_ptr[thread_id] = static_cast<int>(thread_id);
    }
};

TEST_CASE("queueCollective", "[queue]")
{
    // Define the index domain
    using Dim = alpaka::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = alpaka::Dev<Acc>;

    using Queue = alpaka::QueueCpuOmp2Collective;
    using Pltf = alpaka::Pltf<Dev>;

    auto dev = alpaka::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    using Vec = alpaka::Vec<Dim, Idx>;
    Vec const elements_per_thread(Vec::all(static_cast<Idx>(1)));
    Vec const threads_per_block(Vec::all(static_cast<Idx>(1)));
    Vec const blocks_per_grid(results.size());

    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    WorkDiv const work_div(blocks_per_grid, threads_per_block, elements_per_thread);

#    pragma omp parallel num_threads(static_cast <int>(results.size()))
    {
        // The kernel will be performed collectively.
        // OpenMP will distribute the work between the threads from the parallel region
        alpaka::exec<Acc>(queue, work_div, QueueCollectiveTestKernel{}, results.data());

        alpaka::wait(queue);
    }

    for(size_t i = 0; i < results.size(); ++i)
    {
        REQUIRE(static_cast<int>(i) == results.at(i));
    }
}

TEST_CASE("TestCollectiveMemcpy", "[queue]")
{
    // Define the index domain
    using Dim = alpaka::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = alpaka::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = alpaka::Dev<Acc>;

    using Queue = alpaka::QueueCpuOmp2Collective;
    using Pltf = alpaka::Pltf<Dev>;

    auto dev = alpaka::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    // Define the work division
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec const elements_per_thread(Vec::all(static_cast<Idx>(1)));
    Vec const threads_per_block(Vec::all(static_cast<Idx>(1)));
    Vec const blocks_per_grid(results.size());

    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    WorkDiv const work_div(blocks_per_grid, threads_per_block, elements_per_thread);

#    pragma omp parallel num_threads(static_cast <int>(results.size()))
    {
        int thread_id = omp_get_thread_num();

        using View = alpaka::ViewPlainPtr<Dev, int, Dim, Idx>;

        View dst(results.data() + thread_id, dev, Vec(static_cast<Idx>(1u)), Vec(sizeof(int)));

        View src(&thread_id, dev, Vec(static_cast<Idx>(1u)), Vec(sizeof(int)));

        // avoid that the first thread is executing the copy (can not be guaranteed)
        size_t sleep_ms = (results.size() - static_cast<uint32_t>(thread_id)) * 100u;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));

        // only one thread will perform this memcpy
        alpaka::memcpy(queue, dst, src, Vec(static_cast<Idx>(1u)));

        alpaka::wait(queue);
    }

    uint32_t num_flipped_values = 0u;
    uint32_t num_non_intitial_values = 0u;
    for(size_t i = 0; i < results.size(); ++i)
    {
        if(static_cast<int>(i) == results.at(i))
            num_flipped_values++;
        if(results.at(i) != -1)
            num_non_intitial_values++;
    }
    // only one thread is allowed to flip the value
    REQUIRE(num_flipped_values == 1u);
    // only one value is allowed to differ from the initial value
    REQUIRE(num_non_intitial_values == 1u);
}

#endif
