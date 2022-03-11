// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream<i32>

    sdfg.state @state_0 {        
        %1 = sdfg.tasklet @one() -> i32{
                %1 = arith.constant 1 : i32
                sdfg.return %1 : i32
            }

        sdfg.stream_push %1, %A : i32 -> !sdfg.stream<i32>
        %a_1 = sdfg.stream_pop %A : !sdfg.stream<i32> -> i32
    }
}