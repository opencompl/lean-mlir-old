---
title: "lean-mlir: Operations"
---

[Back to index](index.html)

## Specification of operations

The main information that we store about dialects is the operations that they define. Each of these is specified (in its abstract form) by the following parameters:

* A **fully-qualified non-unique name** (eg `toy.transpose`). Having two instructions with the same name is allowed and called **overloading**. The following properties are specific to each overload.
* A **fixed number of arguments** (either *single* or *variadic*), **basic block arguments** and **regions**.
* A **fixed number of return values**.
* A finite set of **allowed attributes** (name and type), with a **requirement specification** (either *required* or *optional*).
* A **matching term** describing some of the constraints.
* Additional **side constraints**.

Concrete operations found in programs are *instances* of this specification; the arguments are actual SSA values (or list thereof) with types set in context, basic blocks defined elsewhere in the program, they have fully-specified attributes, etc.

* <span style='color:#d04040'>**(Important)**</span> Define arbitrary constraints (for IRDL)
* <span style='color:#d04040'>**(Important)**</span> Support attributes in `MTerm` operations
* <span style='color:#ce9d09'>**(Useful)**</span> Traits
* <span style='color:#8f8f8f'>**(Misc)**</span> Multi-typed attributes?

<span style='text-decoration:underline'>*Example #1: `toy.constant`*</span>

```
%t = "toy.constant"() {value=dense<[[1,2],[3,4]]>: tensor<2x2xi32>}:
  () -> tensor<2x2xi32>
```

* Name: `toy.constant`
* Arguments: no arguments, no basic block arguments, no regions
* Return values: 1
* Allowed attributes: `value` (required)
* Side constraints: none

Generic form as a matching term:

```
OP ["toy.constant",
    LIST [],
    LIST [OPERAND [%ret, TENSOR [$D, !τ]]
    # TODO: Attributes in MTerm operations
]
```

<span style='text-decoration:underline'>*Example #2: `arith.addi` (first overload)*</span>

```
%z = "arith.addi"(%x, %y): (i32, i32) -> i32
```

* Name: `arith.addi`
* Arguments: 2 single arguments, no basic block arguments, no regions
* Return values: 1
* Allowed attributes: none
* Side constraints: none

Generic form as a unifiable term (overload #1):

```
OP ["toy.addi",
    LIST [OPERAND [%x, INT [$sgn, $sz]],
          OPERAND [%y, INT [$sgn, $sz]]],
    LIST [OPERAND [%z, INT [$sgn, $sz]]]
]
```

<span style='text-decoration:underline'>*Example #3: `arith.addi` (second overload)*</span>

```
%z = "arith.addi"(%x, %y):
     (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
```

* Name: `arith.addi`
* Arguments: 2 single arguments, no basic block arguments, no regions
* Return values: 1
* Allowed attributes: none
* Side constraints: none

Generic form as a unifiable term (overload #2):

```
OP ["toy.addi",
    LIST [OPERAND [%x, TENSOR [$D, INT [$sgn, $sz]]],
          OPERAND [%y, TENSOR [$D, INT [$sgn, $sz]]]],
    LIST [OPERAND [%z, TENSOR [$D, INT [$sgn, $sz]]]]
]
```

