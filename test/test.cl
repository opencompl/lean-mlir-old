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
(defparameter *llvm-project-path* #P"/home/bollu/work/llvm-project/")
(defparameter *tests*
  (list
    (list "IR"  (merge-pathnames #P"mlir/test/IR/*.mlir" *llvm-project-path*))
    (list "AMX"  (merge-pathnames #P"mlir/test/Dialect/AMX/*.mlir" *llvm-project-path*))
    (list "Affine" (merge-pathnames #P"mlir/test/Dialect/Affine/*" *llvm-project-path*))
    (list "Arith"  (merge-pathnames #P"mlir/test/Dialect/Arithmetic/*" *llvm-project-path*))
    (list "ArmNeon"  (merge-pathnames #P"mlir/test/Dialect/ArmNeon/*" *llvm-project-path*))
    (list "ArmSVE"  (merge-pathnames #P"mlir/test/Dialect/ArmSVE//*" *llvm-project-path*))
    (list "Async"  (merge-pathnames #P"mlir/test/Dialect/Async/*" *llvm-project-path*))
    (list "Bufferization" (merge-pathnames #P"mlir/test/Dialect/Bufferization/*" *llvm-project-path*))
    (list "Builtin"  (merge-pathnames #P"mlir/test/Dialect/Builtin/*" *llvm-project-path*))
    (list "Complex"  (merge-pathnames #P"mlir/test/Dialect/Complex/*" *llvm-project-path*))
    (list "ControlFlow"  (merge-pathnames #P"mlir/test/Dialect/ControlFlow/*" *llvm-project-path*))
    (list "DTLI"  (merge-pathnames #P"mlir/test/Dialect/DLTI/*" *llvm-project-path*))
    (list "EmitC"  (merge-pathnames #P"mlir/test/Dialect/EmitC/*" *llvm-project-path*))
    (list "Func"  (merge-pathnames #P"mlir/test/Dialect/Func/*" *llvm-project-path*))
    (list "GPU"  (merge-pathnames #P"mlir/test/Dialect/GPU/*" *llvm-project-path*))
    (list "LLVMIR"  (merge-pathnames #P"mlir/test/Dialect/LLVMIR/*" *llvm-project-path*))
    (list "Linalg"  (merge-pathnames #P"mlir/test/Dialect/Linalg/*" *llvm-project-path*))
    (list "Math"  (merge-pathnames #P"mlir/test/Dialect/Math" *llvm-project-path*))
    (list "MemRef"  (merge-pathnames #P"mlir/test/Dialect/MemRef/*" *llvm-project-path*))
    (list "OpenACC"  (merge-pathnames #P"mlir/test/Dialect/OpenACC/*" *llvm-project-path*))
    (list "OpenMP"  (merge-pathnames #P"mlir/test/Dialect/OpenMP/*" *llvm-project-path*))
    (list "PDL"  (merge-pathnames #P"mlir/test/Dialect/PDL/*" *llvm-project-path*))
    (list "PDLInterp"  (merge-pathnames #P"mlir/test/Dialect/PDLInterp/*" *llvm-project-path*))
    (list "Quant"  (merge-pathnames #P"mlir/test/Dialect/Quant/*" *llvm-project-path*))
    (list "SPIRV"  (merge-pathnames #P"mlir/test/Dialect/SPIRV/*" *llvm-project-path*))
    (list "Shape"  (merge-pathnames #P"mlir/test/Dialect/Shape/*" *llvm-project-path*))
    (list "SparseTensor"  (merge-pathnames #P"mlir/test/Dialect/SparseTensor/*" *llvm-project-path*))
    (list "Tensor"  (merge-pathnames #P"mlir/test/Dialect/Tensor/*.mlir" *llvm-project-path*))
    (list "Tosa"  (merge-pathnames #P"mlir/test/Dialect/Tosa/*" *llvm-project-path*))
    (list "Vector"  (merge-pathnames #P"mlir/test/Dialect/Vector/*" *llvm-project-path*))
    (list "X86Vector"  (merge-pathnames #P"mlir/test/Dialect/X86Vector/*")) *llvm-project-path*))

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

