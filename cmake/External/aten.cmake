if (NOT __ATEN_INCLUDED)
  set(__ATEN_INCLUDED TRUE)

    # build directory
    set(ATEN_PREFIX ${CMAKE_BINARY_DIR}/external/aten-prefix)  
    # install directory
    set(ATEN_INSTALL ${CMAKE_BINARY_DIR}/external/tmp_install)

    set(ATEN_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${ATEN_EXTRA_COMPILER_FLAGS})
    set(ATEN_C_FLAGS ${CMAKE_C_FLAGS} ${ATEN_EXTRA_COMPILER_FLAGS})

    # depend on gflags if we're also building it
    if (ATEN_EXTERNAL)
      set(ATEN_DEPENDS aten)
    endif()

    ExternalProject_Add(aten
      PREFIX ${ATEN_PREFIX}
      SOURCE_DIR  "${PROJECT_SOURCE_DIR}/torch/aten"
      INSTALL_DIR ${ATEN_INSTALL}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_INSTALL_PREFIX=${ATEN_INSTALL}
                 -DCMAKE_C_FLAGS=${ATEN_C_FLAGS}
                 -DCMAKE_CXX_FLAGS=${ATEN_CXX_FLAGS}
      )

    set(ATEN_FOUND TRUE)
    set(ATEN_INCLUDE_DIRS ${ATEN_INSTALL}/include)
    set(ATEN_LIBRARIES ${ATENS_LIBRARIES} ${ATEN_INSTALL}/lib/libATen.dll.a)
    set(ATEN_LIBRARY_DIRS ${ATEN_INSTALL}/lib)
    set(ATEN_EXTERNAL TRUE)

    list(APPEND external_project_dependencies aten)
    
    INSTALL(FILES  ${CMAKE_BINARY_DIR}/external/tmp_install/lib/libATen.dll.a  DESTINATION lib)
    INSTALL(FILES  ${CMAKE_BINARY_DIR}/external/tmp_install/bin/libATen.dll  DESTINATION lib)

endif()

