// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg 
// CHECK-SAME: [[SDFG0:@[a-zA-Z0-9_]*]]
// CHECK-SAME: ({{%[a-zA-Z0-9_]*}}: !sdir.array<i32>, {{%[a-zA-Z0-9_]*}}: !sdir.array<i32>) -> !sdir.array<i32>
sdir.sdfg{entry=@state_0} @sdfg_0(%a: !sdir.array<i32>, %b: !sdir.array<i32>) -> !sdir.array<i32> {
    // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
    sdir.state @state_0 {
        // CHECK: sdir.sdfg
        sdir.sdfg{entry=@state_1} @sdfg_1 {
            // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
            sdir.state @state_1 {
                // CHECK: [[ARRAYN:%[a-zA-Z0-9_]*]] = sdir.alloc()
                // CHECK-SAME: !sdir.array<i32>
                %N = sdir.alloc() : !sdir.array<i32>
                // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.call [[SDFG0]]
                // CHECK-SAME: ([[ARRAYN]], [[ARRAYN]])
                // CHECK-SAME: (!sdir.array<i32>, !sdir.array<i32>) -> !sdir.array<i32>
                %c = sdir.call @sdfg_0(%N, %N) : (!sdir.array<i32>, !sdir.array<i32>) -> !sdir.array<i32>
            }
        }

    }
} 
