;;; -*- Mode: Common-Lisp; Author: Siddharth-Bhat -*-
;;;; Version 1:
;;;; ---------
;;;; Keep the project at ~/quicklisp/local-projects/hoopl.asdl. Run M-x slime. Follow with (ql:quickload hoopl) in REPL. Finally
;;;; switch to hoopl.lisp and type C-c ~ [slime-sync-package-and-default-directory]
;;;; Version 2:
;;;; ---------
;;;; Open hoopl.asd, run C-c C-k [compile file]. Switch to REPL, then run (ql:quickload hoopl). Finally switch to hoopl.lisp
;;;; and type C-c ~ [slime-sync-package-and-default-directory] to enter the hoopl module in the repl

;; (in-package :hoopl)
;; (sb-ext:restrict-compiler-policy 'debug 3 3)
(declaim (optimize (speed 0) (space 0) (debug 3)))

;; parinfer and rainbow delimiters interact poorly!
;; :(
(defparameter *tests*
  '(
    ("IR"  #P"/home/siddu_druid/work/llvm-project/mlir/test/IR/*.mlir")
    ("AMX"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/AMX/*.mlir")
    ("Affine" #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Affine/*")
    ("Arith"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Arithmetic/*")
    ("ArmNeon"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/ArmNeon/*")
    ("ArmSVE"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/ArmSVE//*")
    ("Async"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Async/*")
    ("Bufferization" #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Bufferization/*")
    ("Builtin"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Builtin/*")
    ("Complex"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Complex/*")
    ("ControlFlow"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/ControlFlow/*")
    ("DTLI"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/DLTI/*")
    ("EmitC"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/EmitC/*")
    ("Func"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Func/*")
    ("GPU"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/GPU/*")
    ("LLVMIR"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/LLVMIR/*")
    ("Linalg"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Linalg/*")
    ("Math"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Math")
    ("MemRef"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/MemRef/*")
    ("OpenACC"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/OpenACC/*")
    ("OpenMP"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/OpenMP/*")
    ("PDL"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/PDL/*")
    ("PDLInterp"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/PDLInterp/*")
    ("Quant"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Quant/*")
    ("SPIRV"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/SPIRV/*")
    ("Shape"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Shape/*")
    ("SparseTensor"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/SparseTensor/*")
    ("Tensor"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Tensor/*.mlir")
    ("Tosa"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Tosa/*")
    ("Vector"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/Vector/*")
    ("X86Vector"  #P"/home/siddu_druid/work/llvm-project/mlir/test/Dialect/X86Vector/*")))

(defparameter *total* 0)

(defun canonicalize-program-str (p nsucc)
  (multiple-value-bind (out err retval)
	(uiop:run-program `("mlir-opt"
			    ,(namestring p)
			    "--mlir-print-op-generic"
			    "--allow-unregistered-dialect")
			  :output :string :error-output :string)
    (declare (ignore out))
    (declare (ignore err))
    (declare (ignore retval))
    (incf nsucc)))




(defun main ()
  (let ((nsucc 0))
    (loop for (name path-root) in *tests* do
      (format t "running on path |~d|~%" path-root)
      (let* ((ps (directory path-root)))
	(format t "...found |~d| files~%" (length ps))
	(loop for p in ps do
	     (canonicalize-program-str p nsucc)
	     (format t "num-successes in ~d: ~d/~d~%"
		     path-root nsucc (length ps))
	      )))))

(main)

