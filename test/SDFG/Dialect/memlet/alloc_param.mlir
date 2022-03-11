// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {

    sdfg.state @state_0{
        // CHECK: [[NAMEn:%[a-zA-Z0-9_]*]] = sdfg.tasklet @five
        %n = sdfg.tasklet @five() -> index{
            %5 = arith.constant 5 : index
            sdfg.return %5 : index
        }
        // CHECK: [[NAMEm:%[a-zA-Z0-9_]*]] = sdfg.tasklet @twenty
        %m = sdfg.tasklet @twenty() -> index{
            %20 = arith.constant 20 : index
            sdfg.return %20 : index
        }
        // CHECK: {{%[a-zA-Z0-9_]*}} = sdfg.alloc([[NAMEn]], [[NAMEm]])
        // CHECK-SAME: !sdfg.array<?x?xi32>
        %a = sdfg.alloc(%n, %m) : !sdfg.array<?x?xi32>
    }
}