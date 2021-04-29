/* Copyright 2021 Bernhard Manfred Gruber

 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma once

#include <alpaka/core/Utility.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/meta/DependentFalseType.hpp>
#include <alpaka/meta/TypeListOps.hpp>

#include <tuple>

namespace alpaka
{
    //! Access tag type indicating read-only access.
    struct ReadAccess
    {
    };

    //! Access tag type indicating write-only access.
    struct WriteAccess
    {
    };

    //! Access tag type indicating read-write access.
    struct ReadWriteAccess
    {
    };

    //! An accessor is an abstraction for accessing memory objects such as views and buffers.
    //! @tparam TAcc The accelerator on which this accessor is used.
    //! @tparam TElem The type of the element stored by the memory object. Values and references to this type are
    //! returned on access.
    //! @tparam TBufferIdx The integral type used for indexing and index computations.
    //! @tparam TDim The dimensionality of the accessed data.
    //! @tparam TAccessModes A sequence of access tag types.
    template<typename TAcc, typename TElem, typename TBufferIdx, std::size_t TDim, typename... TAccessModes>
    struct Accessor;

    //! Accessors with multiple access modes, by default, behave as an accessor of their first access mode.
    template<
        typename TAcc,
        typename TElem,
        typename TBufferIdx,
        std::size_t TDim,
        typename TAccessMode1,
        typename TAccessMode2,
        typename... TAccessModes>
    struct Accessor<TAcc, TElem, TBufferIdx, TDim, TAccessMode1, TAccessMode2, TAccessModes...>
        : Accessor<TAcc, TElem, TBufferIdx, TDim, TAccessMode1>
    {
        using Accessor<TAcc, TElem, TBufferIdx, TDim, TAccessMode1>::Accessor;
    };

    namespace traits
    {
        //! The customization point for how to build an accessor for a given accelerator.
        template<typename TAcc, typename SFINAE = void>
        struct BuildAccessor;
    } // namespace traits

    namespace internal
    {
        template<typename T>
        struct IsAccessor : std::false_type
        {
        };

        template<typename TAcc, typename TElem, typename TBufferIdx, std::size_t Dim, typename... TAccessModes>
        struct IsAccessor<Accessor<TAcc, TElem, TBufferIdx, Dim, TAccessModes...>> : std::true_type
        {
        };
    } // namespace internal

    //! Creates an accessor for the given memory object using the specified access modes. Memory objects are e.g.
    //! alpaka views and buffers.
    template<
        typename TAcc,
        typename... TAccessModes,
        typename TMemoryObject,
        typename = std::enable_if_t<!internal::IsAccessor<std::decay_t<TMemoryObject>>::value>>
    ALPAKA_FN_HOST_ACC auto accessWith(TMemoryObject&& memoryObject)
    {
        return traits::BuildAccessor<TAcc>::template buildAccessor<TAccessModes...>(memoryObject);
    }

    //! Constrains an existing accessor with multiple access modes to the specified access modes.
    // TODO: currently only allows constraining down to 1 access mode
    template<
        typename TAcc,
        typename TNewAccessMode,
        typename TElem,
        typename TBufferIdx,
        std::size_t TDim,
        typename TPrevAccessMode1,
        typename TPrevAccessMode2,
        typename... TPrevAccessModes>
    ALPAKA_FN_HOST_ACC auto accessWith(
        const Accessor<TAcc, TElem, TBufferIdx, TDim, TPrevAccessMode1, TPrevAccessMode2, TPrevAccessModes...>& acc)
    {
        static_assert(
            meta::Contains<std::tuple<TPrevAccessMode1, TPrevAccessMode2, TPrevAccessModes...>, TNewAccessMode>::value,
            "The accessed accessor must already contain the requested access mode");
        return Accessor<TAcc, TElem, TBufferIdx, TDim, TNewAccessMode>{acc};
    }

    //! Constrains an existing accessor to the specified access modes.
    // constraining accessor to the same access mode again just passes through
    template<typename TAcc, typename TNewAccessMode, typename TElem, typename TBufferIdx, std::size_t TDim>
    ALPAKA_FN_HOST_ACC auto accessWith(const Accessor<TAcc, TElem, TBufferIdx, TDim, TNewAccessMode>& acc)
    {
        return acc;
    }

    //! Creates a read-write accessor for the given memory object (view, buffer, ...).
    template<typename TAcc, typename TMemoryObject>
    ALPAKA_FN_HOST_ACC auto access(TMemoryObject&& view)
    {
        return accessWith<TAcc, ReadWriteAccess>(std::forward<TMemoryObject>(view));
    }

    //! Constrains an existing accessor to read/write access.
    template<typename TAcc, typename TElem, typename TBufferIdx, std::size_t TDim, typename... TAccessModes>
    ALPAKA_FN_HOST_ACC auto access(const Accessor<TAcc, TElem, TBufferIdx, TDim, TAccessModes...>& acc)
    {
        return accessWith<TAcc, ReadWriteAccess>(acc);
    }

    //! Creates a read-only accessor for the given memory object (view, buffer, ...).
    template<typename TAcc, typename TMemoryObjectOrAccessor>
    ALPAKA_FN_HOST_ACC auto readAccess(TMemoryObjectOrAccessor&& viewOrAccessor)
    {
        return accessWith<TAcc, ReadAccess>(std::forward<TMemoryObjectOrAccessor>(viewOrAccessor));
    }

    //! Constrains an existing accessor to read access.
    template<typename TAcc, typename TElem, typename TBufferIdx, std::size_t TDim, typename... TAccessModes>
    ALPAKA_FN_HOST_ACC auto readAccess(const Accessor<TAcc, TElem, TBufferIdx, TDim, TAccessModes...>& acc)
    {
        return accessWith<TAcc, ReadAccess>(acc);
    }

    //! Creates a write-only accessor for the given memory object (view, buffer, ...).
    template<typename TAcc, typename TMemoryObject>
    ALPAKA_FN_HOST_ACC auto writeAccess(TMemoryObject&& view)
    {
        return accessWith<TAcc, WriteAccess>(std::forward<TMemoryObject>(view));
    }

    //! Constrains an existing accessor to write access.
    template<typename TAcc, typename TElem, typename TBufferIdx, std::size_t TDim, typename... TAccessModes>
    ALPAKA_FN_HOST_ACC auto writeAccess(const Accessor<TAcc, TElem, TBufferIdx, TDim, TAccessModes...>& acc)
    {
        return accessWith<TAcc, WriteAccess>(acc);
    }
} // namespace alpaka
