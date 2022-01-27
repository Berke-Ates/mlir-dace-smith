// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.stream_array<i32>
    %A = sdir.alloc() : !sdir.stream_array<i32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.stream_array<i32> -> !sdir.stream<i32>
        %a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.stream<i32>
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.stream_pop [[NAMEB]]
        // CHECK-SAME: !sdir.stream<i32> -> i32
        %a_1 = sdir.stream_pop %a : !sdir.stream<i32> -> i32
    }
}
