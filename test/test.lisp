;; -*- Mode: Common-Lisp; Author: Siddharth-Bhat -*-;
;;; Version 1:
;;;; ---------
;;;; Keep the project at ~/quicklisp/local-projects/hoopl.asdl. Run M-x slime. Follow with (ql:quickload hoopl) in REPL. Finally
;;;; switch to hoopl.lisp and type C-c ~ [slime-sync-package-and-default-directory]
;;;; Version 2:
;;;; ---------
;;;; Open hoopl.asd, run C-c C-k [compile file]. Switch to REPL, then run (ql:quickload hoopl). Finally switch to hoopl.lisp
;;;; and type C-c ~ [slime-sync-package-and-default-directory] to enter the hoopl module in the repl

;; (in-package :hoopl)
;; (sb-ext:restrict-compiler-policy 'debug 3 3)
;;(ql:quickload `clouseau) ;; lisp inspector with (clouseau:inspect ...)
(ql:quickload 'str)
(ql:quickload 'iterate)
(declaim (optimize (speed 0) (space 0) (debug 3)))

;; parinfer and rainbow delimiters interact poorly!
;; :(
(defparameter *llvm-project-path* #P"/home/bollu/work/llvm-project/")


(defparameter *tests*
  (list
    (list "IR"  (merge-pathnames #P"mlir/test/IR/*.mlir" *llvm-project-path*))
    (list "AMX"  (merge-pathnames #P"mlir/test/Dialect/AMX/*.mlir" *llvm-project-path*))
    (list "Affine" (merge-pathnames #P"mlir/test/Dialect/Affine/*.mlir" *llvm-project-path*))
    (list "Arith"  (merge-pathnames #P"mlir/test/Dialect/Arithmetic/*.mlir" *llvm-project-path*))
    (list "ArmNeon"  (merge-pathnames #P"mlir/test/Dialect/ArmNeon/*.mlir" *llvm-project-path*))
    (list "ArmSVE"  (merge-pathnames #P"mlir/test/Dialect/ArmSVE/*.mlir" *llvm-project-path*))
    (list "Async"  (merge-pathnames #P"mlir/test/Dialect/Async/*.mlir" *llvm-project-path*))
    (list "Bufferization" (merge-pathnames #P"mlir/test/Dialect/Bufferization/*.mlir" *llvm-project-path*))
    (list "Builtin"  (merge-pathnames #P"mlir/test/Dialect/Builtin/*.mlir" *llvm-project-path*))
    (list "Complex"  (merge-pathnames #P"mlir/test/Dialect/Complex/*.mlir" *llvm-project-path*))
    (list "ControlFlow"  (merge-pathnames #P"mlir/test/Dialect/ControlFlow/*.mlir" *llvm-project-path*))
    (list "DTLI"  (merge-pathnames #P"mlir/test/Dialect/DLTI/*.mlir" *llvm-project-path*))
    (list "EmitC"  (merge-pathnames #P"mlir/test/Dialect/EmitC/*.mlir" *llvm-project-path*))
    (list "Func"  (merge-pathnames #P"mlir/test/Dialect/Func/*.mlir" *llvm-project-path*))
    (list "GPU"  (merge-pathnames #P"mlir/test/Dialect/GPU/*.mlir" *llvm-project-path*))
    (list "LLVMIR"  (merge-pathnames #P"mlir/test/Dialect/LLVMIR/*.mlir" *llvm-project-path*))
    (list "Linalg"  (merge-pathnames #P"mlir/test/Dialect/Linalg/*.mlir" *llvm-project-path*))
    (list "Math"  (merge-pathnames #P"mlir/test/Dialect/Math/*.mlir" *llvm-project-path*))
    (list "MemRef"  (merge-pathnames #P"mlir/test/Dialect/MemRef/*.mlir" *llvm-project-path*))
    (list "OpenACC"  (merge-pathnames #P"mlir/test/Dialect/OpenACC/*.mlir" *llvm-project-path*))
    (list "OpenMP"  (merge-pathnames #P"mlir/test/Dialect/OpenMP/*.mlir" *llvm-project-path*))
    (list "PDL"  (merge-pathnames #P"mlir/test/Dialect/PDL/*.mlir" *llvm-project-path*))
    (list "PDLInterp"  (merge-pathnames #P"mlir/test/Dialect/PDLInterp/*.mlir" *llvm-project-path*))
    (list "Quant"  (merge-pathnames #P"mlir/test/Dialect/Quant/*.mlir" *llvm-project-path*))
    (list "SPIRV"  (merge-pathnames #P"mlir/test/Dialect/SPIRV/*.mlir" *llvm-project-path*))
    (list "Shape"  (merge-pathnames #P"mlir/test/Dialect/Shape/*.mlir" *llvm-project-path*))
    (list "SparseTensor"  (merge-pathnames #P"mlir/test/Dialect/SparseTensor/*.mlir" *llvm-project-path*))
    (list "Tensor"  (merge-pathnames #P"mlir/test/Dialect/Tensor/*.mlir" *llvm-project-path*))
    (list "Tosa"  (merge-pathnames #P"mlir/test/Dialect/Tosa/.mlir*" *llvm-project-path*))
    (list "Vector"  (merge-pathnames #P"mlir/test/Dialect/Vector/*.mlir" *llvm-project-path*))
    (list "X86Vector"  (merge-pathnames #P"mlir/test/Dialect/X86Vector/*.mlir" *llvm-project-path*))))

;; (defparameter *tests* (list (list "ALL" (merge-pathnames #P"mlir/test/**/*.mlir" *llvm-project-path*))))
 
(defparameter *total* 0)

;; returns (error? <output>)
(defun canonicalize-program-str (path)
  (multiple-value-bind (out err retval)
      (uiop:run-program (list "mlir-opt"
			      (namestring path)
			      "--mlir-print-op-generic"
			      "--allow-unregistered-dialect")
			:output :string
			:error-output :string
			:ignore-error-status t)
    (if (/= retval 0)
	;; then: error
	(list retval err)
	;; else: output
	(list retval out))))


;; a canonicalized part of an MLIR file
;; contents will be MLIR generic.
(defstruct mlir-file-part
  path
  split-index
  contents)

;; returns file
;; https://github.com/llvm/llvm-project/blob/860eabb3953a104b2b85398c705cd32e33d86689/mlir/lib/Support/ToolUtilities.cpp#L22
;; mlir::splitAndProcessBuffer
;; 
;; returns a list of mlir-file-part from a file path
(defun read-mlir-file (file-path)
  (declare (type pathname file-path))
  (let* ((contents (uiop:read-file-string file-path))
         ;; parts: list of parts
         (parts (if (search "-split-input-file" contents)
                     (remove-if #'str:emptyp (mapcar #'str:trim (str:split "// -----" contents)))
                     (list contents))))
    (loop for i from 0 for part in parts do 
          (make-mlir-file-part :path file-path :split-index i :contents part))))

(defun main ()
  (let ((nsucc 0) (nfail 0) (out-parts nil))
    (iter:iterate (iter:for (name path-root) in *tests*)
          (format t "collecting path |~d|~%" path-root)
          (let* ((paths (directory path-root)))
            (format t "...found |~d| files~%" (length paths))
            (iter:iterate (iter:for path in paths)
                          (format t "reading path |~d|~%" path)
                          (nconc out-parts (read-mlir-file path)))))
    (iter:iterate (iter:for part in out-parts)
                  (format t "processing |~d:~d|~%" 
                          (mlir-file-part-path part)
                          (mlir-file-part-split-index part))
                  )))
(main)


