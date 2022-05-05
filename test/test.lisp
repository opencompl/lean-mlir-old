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
(ql:quickload 'shasht)
(ql:quickload 'str)
(ql:quickload 'lparallel)
(ql:quickload 'bordeaux-threads)
(ql:quickload 'cl-cram)

(declaim (optimize (speed 0) (space 0) (debug 3)))

;; parinfer and rainbow delimiters interact poorly!
;; :(
;; (defparameter *llvm-project-path* #P"/home/bollu/work/frontend/llvm-project/")
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
   (list "X86Vector"  (merge-pathnames #P"mlir/test/Dialect/X86Vector/*.mlir" *llvm-project-path*))
   ))

;; (defparameter *tests* (list (list "ALL" (merge-pathnames #P"mlir/test/**/*.mlir" *llvm-project-path*))))

(defparameter *total* 0)


;; If canonicalization succeeds, then print canonical form.
;; If not, skip file.
;; (defun canonicalize-program-str (s)
;;   (declare (type string s))
;;   ;; TODO: create a temporary path, write data to file, run mlir-opt on it.
;;   (abort)
;;   (multiple-value-bind (out err retval)
;;       (uiop:run-program (list "mlir-opt"
;; 			      (namestring path)
;; 			      "--mlir-print-op-generic"
;; 			      "--allow-unregistered-dialect")
;; 			:output :string
;; 			:error-output :string
;; 			:ignore-error-status t)
;;     (if (/= retval 0)
;; 	;; then: error
;; 	(list retval err)
;; 	;; else: output
;; 	(list retval out))))

;; a canonicalized part of an MLIR file
;; contents will be MLIR generic.
(defstruct mlir-file-part
  guid ;; global unique ID
  category ;; category of file (SPIRV, Shape, etc.)
  path ;; path of file
  partix ;; partix of data
  contents ;; contents of file
  canon-contents ;; canonicalized contents of file
  canon-error ;; error when trying to canonicalize file
  )

;; returns file
;; https://github.com/llvm/llvm-project/blob/860eabb3953a104b2b85398c705cd32e33d86689/mlir/lib/Support/ToolUtilities.cpp#L22
;; mlir::splitAndProcessBuffer
;; 
;; returns a list of mlir-file-part from a file path
(defparameter *mlir-file-part-guid* 0)
(defun read-mlir-file (category file-path)
  (declare (type pathname file-path))
  (let* ((contents (uiop:read-file-string file-path))
         ;; parts: list of parts
         (parts (if (search "-split-input-file" contents)
		    (remove-if #'str:emptyp
			       (mapcar #'str:trim (str:split "// -----" contents)))
		    (list contents))))
    (format t "......found ~d parts ~%" (length parts))
    (loop for i from 0 for part in parts 
          collect (make-mlir-file-part
		   :guid (incf *mlir-file-part-guid*)
		   :category category
		   :path file-path
		   :partix i
		   :contents part))))



;; statistics associated to each folder
(defstruct stats
  nsucc-run
  nfail-run)

;; default stats (success/fail at zero)
(defun make-default-stats ()  (make-stats :nsucc-run 0 :nfail-run 0))


;; lock for running
(defparameter *run-lock* (bordeaux-threads:make-lock))

;; statistics per content.
(defparameter *run-stats* (make-hash-table :test 'equal))

;; number of files that lake builded successfully.
(defparameter *nsucc-run* 0)
;; number of files that failed a lake build.
(defparameter *nfail-run* 0)


;; global index of the file path, which is used
;; to generate a sequence
(defparameter *lean-file-path-ix* 0)

(defparameter *lean-file-template*
"
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open Lean
open Lean.Parser
open  MLIR.EDSL
open MLIR.AST
open MLIR.Doc
open IO

set_option maxHeartbeats 999999999


declare_syntax_cat mlir_ops
syntax (ws mlir_op ws)* : mlir_ops
syntax \"[mlir_ops|\" mlir_ops \"]\" : term

macro_rules
|`([mlir_ops| $[ $xs ]* ]) => do 
  let xs <- xs.mapM (fun x =>`([mlir_op| $x]))
  quoteMList xs.toList (<- `(MLIR.AST.Op))

  
-- | write an op into the path
def o: List Op := [mlir_ops|
~d
] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile \"~d\" str
")

			 

;; make the name of the lean file.
;; basepath: "1-failure/"
;; extension: "lean"
(defun mlir-file-part-make-filepath (part basepath extension)
  (assert (not (str:contains? "." extension)))
  (let* ((p (mlir-file-part-path part)) ; path
	 (dir (directory-namestring p)) ; directory
	 (fname (pathname-name p)) ; filename 
	 (relative-dir (enough-namestring dir *llvm-project-path*)) ; relative to llvm path
	 (outname-raw (str:concat
		       (write-to-string (mlir-file-part-guid part))
		       "-"
		       relative-dir
		       fname
		       "-"
		       (write-to-string (mlir-file-part-partix part))
		       "."
		       extension)) ; full path
	 (outname-mangle (str:replace-all "/" "Z" outname-raw)) ; mangled
	 (outpath (pathname (str:concat basepath "/" outname-mangle))))
    (ensure-directories-exist outpath) ;; eek, side effects!
    outpath))

;; make the string that should be written ito the lean file
(defun make-lean-file-contents (part)
  (assert (mlir-file-part-canon-contents part))
  (format nil *lean-file-template*
	  (mlir-file-part-canon-contents part)
	  "out.txt"))

;; get value from hash table, ensuring existence
(defun ensure-gethash (key table default-value)
  (multiple-value-bind (value exists-p)
      (gethash key table)
    (if exists-p value (setf (gethash key table) default-value))))

;; creates LEAN file correspnoding to part and compiles/runs it.
;; returns success/failure of compilation.
(defun make-and-run-lean-file (part)
  (let ((outpath (mlir-file-part-make-filepath part "1-gen/" "lean")))    
    (str:to-file outpath (make-lean-file-contents part))
    (setf (uiop:getenv "LEAN_PATH") "../build/lib/")
  (multiple-value-bind (out err retval)
      (uiop:run-program (list "lean" (namestring outpath))
			:output :string
			:error-output :string
			:ignore-error-status t)
    (bordeaux-threads:acquire-lock *run-lock*)
    (let ((s (ensure-gethash (mlir-file-part-category part) *run-stats* (make-default-stats))))
      (if (/= retval 0)
	  ;; vvv error vvv
	  (progn
	    (format t "..ERROR: |~d| failed. error:~%~d~%~d~%"
		    outpath
		    out
		    err)
	    (str:to-file
	     (mlir-file-part-make-filepath part "1-failures/" "lean")
	     (make-lean-file-contents part))
	    (incf (stats-nfail-run s))
	    (incf *nfail-run*))
	  ;; vvv no errror vvv
	  (progn
	    (format t "..SUCCESSS: |~d| succeeded~%" outpath)
	    (incf (stats-nsucc-run s))
	    (incf *nsucc-run*)))))
    (bordeaux-threads:release-lock *run-lock*)
    ))




;; lock for canonicalization phase
(defparameter *canon-lock* (bordeaux-threads:make-lock))

;; canonicalized MLIR parts
(defparameter *canon-mlir-parts* nil)

;; number of files successfully canonicalized
(defparameter *nsucc-canon* 0)

;; number of files failed to canonicalize
(defparameter *nfail-canon* 0)

;; canonializes an MLIR part and sets the slot.
;; returns success / failure of canonicalization.
(defun canonicalize-mlir-part (part)
  (format t "canonicalizing |~d:~d|~%" 
	  (mlir-file-part-path part)
	  (mlir-file-part-partix part))  
  (multiple-value-bind (out err retval)
      (uiop:run-program (list "mlir-opt"
			      (namestring (mlir-file-part-path part))
			      "--mlir-print-op-generic"
			      "--allow-unregistered-dialect")
			:output :string
			:error-output :string
			:ignore-error-status t)
    (setf (mlir-file-part-canon-error part) err)
    (setf (mlir-file-part-canon-contents part) out)
    (bordeaux-threads:acquire-lock *canon-lock*)
    (if (= retval 0)
   	(progn
	  (format t "..SUCCESS: canonicalized |~d:~d|~%"
		  (mlir-file-part-path part) (mlir-file-part-partix part))
	  (incf *nsucc-canon*)
	  (push part *canon-mlir-parts*))
	(progn
	  (format t "..ERROR: unable to canonicalize |~d:~d|~%~d"
		  (mlir-file-part-path part)
		  (mlir-file-part-partix part)
		  (mlir-file-part-canon-error part))
	  (incf *nfail-canon*)))
    (bordeaux-threads:release-lock *canon-lock*)
    (when (= retval 0)
      (format t "===[~d/~d]===~%" (mlir-file-part-guid part) (length *raw-mlir-parts*))
      (str:to-file
       (mlir-file-part-make-filepath part "1-canon/" "mlir")
       (mlir-file-part-canon-contents part)))))


;; raw MLIR parts that are read from disk
(defparameter *raw-mlir-parts* nil)

(defun main ()
  (setf lparallel:*kernel* (lparallel:make-kernel 32))
  (loop for (category path-root) in *tests* do
    (format t "collecting category |~d| at  path |~d|~%" category path-root)
    (let* ((paths (directory path-root)))
      (format t "..found |~d| files~%" (length paths))
      (loop for path in paths do
	;; TODO ignore test files
	(format t "....reading path |~d|~%" path)
	(setf *raw-mlir-parts*
	      (append *raw-mlir-parts* (read-mlir-file category path))))))
  (format t "found total |~d| parts |~d| ~%"
	  (length *raw-mlir-parts*) *raw-mlir-parts*)
  ;; try to canonicalize
  (lparallel:pmap nil #'canonicalize-mlir-part *raw-mlir-parts*)
  ;; TODO: reindex the guid based on the new array length after filtering.
  ;; create lean files
  (lparallel:pmap nil #'make-and-run-lean-file *canon-mlir-parts*)
  (format t "hash table: |~d|~%" *run-stats*)
  (loop for k being the hash-keys of *run-stats* using (hash-value v) do
    (format t "~a => (SUCC ~a | FAIL ~a)/~a~%"
	    k
	    (stats-nsucc-run v)
	    (stats-nfail-run v)
	    (+ (stats-nsucc-run v) (stats-nfail-run v))))
  (setf shasht:*write-alist-as-object* t)
  (uiop:with-output-file (outf #P"./test.json" :if-exists :supersede)
    (shasht:write-json (list (cons :stats  *run-stats*)) outf))
  )

(main)



