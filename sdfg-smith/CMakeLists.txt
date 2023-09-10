# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
    ${dialect_libs}
    MLIR_SDFG
    MLIRIR 
    MLIRSmithLib
)

add_mlir_tool(sdfg-smith sdfg-smith.cpp DEPENDS ${LIBS})

target_link_libraries(sdfg-smith PRIVATE ${LIBS})
llvm_update_compile_flags(sdfg-smith)

mlir_check_all_link_libraries(sdfg-smith)

target_sources(SOURCE_FILES_CPP PRIVATE sdfg-smith.cpp)
