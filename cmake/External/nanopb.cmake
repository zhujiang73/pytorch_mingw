if (NOT __NANOPB_INCLUDED)
  set(__NANOPB_INCLUDED TRUE)

    # build directory
    set(NANOPB_PREFIX ${CMAKE_BINARY_DIR}/external/nanopb-prefix)  
    # install directory
    set(NANOPB_INSTALL ${CMAKE_BINARY_DIR}/external/tmp_install)

    set(NANOPB_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${NANOPB_EXTRA_COMPILER_FLAGS})
    set(NANOPB_C_FLAGS ${CMAKE_C_FLAGS} ${NANOPB_EXTRA_COMPILER_FLAGS})

    # depend on gflags if we're also building it
    if (NANOPB_EXTERNAL)
      set(NANOPB_DEPENDS nanopb)
    endif()

    ExternalProject_Add(NANOPB
      PREFIX ${NANOPB_PREFIX}
      SOURCE_DIR  "${PROJECT_SOURCE_DIR}/torch/nanopb"
      INSTALL_DIR ${NANOPB_INSTALL}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_INSTALL_PREFIX=${NANOPB_INSTALL}
                 -DCMAKE_C_FLAGS=${NANOPB_C_FLAGS}
                 -DCMAKE_CXX_FLAGS=${NANOPB_CXX_FLAGS}
      )

    set(NANOPB_FOUND TRUE)
    set(NANOPB_INCLUDE_DIRS ${NANOPB_INSTALL}/include)
    set(NANOPB_LIBRARIES ${NANOPBS_LIBRARIES} ${NANOPB_INSTALL}/lib/libprotobuf-nanopb.a)
    set(NANOPB_LIBRARY_DIRS ${NANOPB_INSTALL}/lib)
    set(NANOPB_EXTERNAL TRUE)

    list(APPEND external_project_dependencies nanopb)
    
    FILE(GLOB_RECURSE NANOPB_H  ${CMAKE_SOURCE_DIR}/torch/libnanopb_windows/*.h)
    INSTALL(FILES  ${NANOPB_H}  DESTINATION include)

    INSTALL(FILES  ${CMAKE_BINARY_DIR}/external/tmp_install/lib/libprotobuf-nanopb.a  DESTINATION lib)
    

endif()

