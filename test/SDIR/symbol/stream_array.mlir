// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state
    sdir.state @state_0{
        // CHECK-NEXT: sdir.alloc_symbol("N")
        sdir.alloc_symbol("N")
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc_stream()
        // CHECK-SAME: !sdir.stream_array<sym("N")xi32>
        %a = sdir.alloc_stream() : !sdir.stream_array<sym("N")xi32>
    }
}
