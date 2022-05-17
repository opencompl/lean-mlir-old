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
(ql:quickload 'cl-csv)


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

(defun log-command-execution (args)
  (format t "sh$ |~d|~%" args)
  args)



;; a canonicalized part of an MLIR file
;; contents will be MLIR generic.
(defstruct mlir-file-part
  guid		   ;; global unique ID
  category	   ;; category of file (SPIRV, Shape, etc.)
  path		   ;; path of file
  partix	   ;; partix of data
  raw-contents	   ;; raw contents of file before caonicalization
  canon-contents   ;; canonicalized contents of file
  canon-error	   ;; error when trying to canonicalize file
  canon-successp   ;; whether canonicalization succeeded
  compile-error	   ;; errors after round tripping
  compile-successp ;; whether compilation succeeded
  run-out	   ;; stdout when running
  run-canon-successp ;; whether we successfully canonicalized the run output
  run-canon-out	     ;; stdout when canonicalizing run output
  run-canon-err	     ; stderr when canonicalizing run output
  roundtrip-successp ; whether the file round-trips correctly
  )



;;  TODO: find some godforsaken way to maintain a connection
;; between the preamble and the data written?
;; Wait, I'm a genius, it's fucking lisp
(defun function-name (fn)
  (string-downcase (symbol-name (nth-value 2 (function-lambda-expression fn)))))

(defparameter *csv-fields-to-write*
  (list #'mlir-file-part-guid
	#'mlir-file-part-category
	#'mlir-file-part-path
	#'mlir-file-part-canon-error
	#'mlir-file-part-canon-successp
	#'mlir-file-part-compile-error
	#'mlir-file-part-compile-successp
	#'mlir-file-part-run-canon-successp
	#'mlir-file-part-run-canon-err
	#'mlir-file-part-roundtrip-successp))



;; Literallyu se the functions as your keys, lol
(defparameter *csv-preamble*
  (loop for f in *csv-fields-to-write*
	collect (str:replace-first "mlir-file-part-" "" (function-name f))))


;;; Format mlir as a CSV.
(defun mlir-part-to-csv-row (p)
  (loop for f in *csv-fields-to-write* collect (cl-csv:format-csv-value (funcall f p))))

;; returns file
;; https://github.com/llvm/llvm-project/blob/860eabb3953a104b2b85398c705cd32e33d86689/mlir/lib/Support/ToolUtilities.cpp#L22
;; mlir::splitAndProcessBuffer
;; 
;; returns a list of mlir-file-part from a file path
(defparameter *mlir-file-part-guid* 0)
(defun read-mlir-file (category file-path)
  (declare (type pathname file-path))
  (let* ((contents (str:from-file file-path))
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
		   :raw-contents part))))



;; statistics associated to each folder
(defstruct stats
  nsucc-canon
  nfail-canon
  nsucc-run
  nfail-run)

;; default stats (success/fail at zero)
(defun make-default-stats ()  (make-stats :nsucc-run 0 :nfail-run 0))


;; lock for running
(defparameter *run-lock* (bordeaux-threads:make-lock))

;; statistics per content.
(defparameter *run-stats* (make-hash-table :test 'equal))


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
    -- def astData := gatherASTData o
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
	 (fname (pathname-name p))	; filename
	 (relative-dir (enough-namestring dir *llvm-project-path*)) ; relative to llvm path
	 (outname-raw (str:concat
		       (write-to-string (mlir-file-part-guid part))
		       "-"
		       relative-dir
		       fname
		       "-"
		       (write-to-string (mlir-file-part-partix part))
		       "."
		       extension))			; full path
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

(defun run-sed (path command)
  (uiop:run-program
   (log-command-execution (list "sed" "-i" "-r" command (namestring path)))
   :output :string))



;;  canonicalize the output of the leanfile.
;;  TODO: move this back in?
;; TODO: can this be refactored to directly set the slots of mlir-file-part?
(defun canonicalize-lean-file-output (part filepath)
  (multiple-value-bind (out err retval)
      (uiop:run-program (log-command-execution
			 (list "mlir-opt" (namestring filepath) "--allow-unregistered-dialect" "--mlir-print-op-generic"))
			:output :string
			:error :string
			:ignore-error-status t)
    (setf (mlir-file-part-run-canon-successp part) (= retval 0))
    (setf (mlir-file-part-run-canon-out part) out)
    (setf (mlir-file-part-run-canon-err part) err)))

;; run lean file corresponding to part <part> stored at filepath <filepath>
;; and canonicalize the corresponding output
(defun run-lean-file (part filepath)
  (let ((outpath (mlir-file-part-make-filepath part "1-runout/" "mlir"))
	(out
	  (uiop:run-program (log-command-execution (list "lean" (namestring filepath) "--run"))
			    :output :string)))
    (str:to-file outpath out)
    (setf (mlir-file-part-run-out part) out)
    (canonicalize-lean-file-output part outpath)
    ;;  file round trips successfuly?
    (setf (mlir-file-part-roundtrip-successp part)
	  (equal (mlir-file-part-run-canon-out part)
		 (mlir-file-part-canon-contents part)))))


;;;  compiles lean file.
(defun compile-lean-file (part filepath)
  (setf (uiop:getenv "LEAN_PATH") "../build/lib/")
  (multiple-value-bind (out err retval)
      (uiop:run-program (log-command-execution (list "lean" (namestring filepath)))
			:output :string
			:error-output :string
			:ignore-error-status t)
    (bordeaux-threads:acquire-lock *run-lock*)
    (let ((s (ensure-gethash (mlir-file-part-category part) *run-stats* (make-default-stats))))
      (if (/= retval 0)
	  ;; vvv error vvv
	  (progn
	    (format t "..ERROR: |lean~d| failed. error:~%~d~%~d~%"
		    filepath
		    out
		    err)
	    (str:to-file
	     (mlir-file-part-make-filepath part "1-failures/" "lean")
	     (make-lean-file-contents part))
	    (incf (stats-nfail-run s)))
	  ;; vvv no errror vvv
	  (progn
	    (format t "..SUCCESSS: |~d| succeeded~%" filepath)
	    (incf (stats-nsucc-run s)))))
    (setf (mlir-file-part-compile-error part) err)
    (setf (mlir-file-part-compile-successp part) (= retval 0))
    (bordeaux-threads:release-lock *run-lock*)))

;; creates LEAN file correspnoding to part and compiles/runs it.
;; returns success/failure of compilation.
(defun make-and-run-lean-file (part)
  (let ((outpath (mlir-file-part-make-filepath part "1-gen/" "lean")))
    (str:to-file outpath (make-lean-file-contents part))
    (compile-lean-file part outpath)
    (when (mlir-file-part-compile-successp part) (run-lean-file part outpath))))


;; lock for canonicalization phase
(defparameter *canon-lock* (bordeaux-threads:make-lock))

;; canonicalized MLIR parts
(defparameter *canon-mlir-parts* nil)


;; number of files successfully canonicalized
(defparameter *nsucc-canon* 0)

;; number of files failed to canonicalize
(defparameter *nfail-canon* 0)

;; raw MLIR parts that are read from disk
(defparameter *raw-mlir-parts* nil)


;; canonializes an MLIR part and sets the slot.
;; returns success / failure of canonicalization.
(defun canonicalize-mlir-part (part)
  (let ((canon-path (mlir-file-part-make-filepath part "1-canon/" "mlir")))
    (format t "===[~d/~d]===~%" (mlir-file-part-guid part) (length *raw-mlir-parts*))
    ;; write the contents into the filex
    (str:to-file canon-path (mlir-file-part-raw-contents part))
    (format t "canonicalizing |~d:~d| at |~d|~%"
	    (mlir-file-part-path part)
	    (mlir-file-part-partix part)
	    canon-path)
    (multiple-value-bind (out err retval)
	(uiop:run-program (log-command-execution
			   (list "mlir-opt"
				 (namestring canon-path)
				 "--mlir-print-op-generic"
				 "--allow-unregistered-dialect"))
			  :output :string
			  :error-output :string
			  :ignore-error-status t)
      (setf (mlir-file-part-canon-error part) err)
      (setf (mlir-file-part-canon-contents part) out)

      (if (= retval 0)
	  (progn
	    (format t "..SUCCESS: canonicalized |~d:~d| at |~d| ~%"
		    (mlir-file-part-path part) (mlir-file-part-partix part)
		    canon-path)
	    (bordeaux-threads:with-lock-held (*canon-lock*)
	      (incf *nsucc-canon*)
	      (push part *canon-mlir-parts*)))
	  (progn
	    (format t "..ERROR: unable to canonicalize |~d:~d| at |~d|~%"
		    (mlir-file-part-path part)
		    (mlir-file-part-partix part)
		    canon-path)
	    (bordeaux-threads:with-lock-held (*canon-lock*)
	      (incf *nfail-canon*))))
      (when (= retval 0)
	(str:to-file canon-path out)
	;; perform further canonicalization
	(run-sed canon-path "s/<([^x>]*)x([^x>]*)>/<\1 × \2>/g")
	(run-sed canon-path "s/<([^x>]*)x([^x>]*)x([^x>]*)>/<\1 × \2 × \3>/g")
	(run-sed canon-path "s/<([^x>]*)x([^x>]*)x([^x>]*)x([^x>]*)>/<\1 × \2 × \3 × \4>/g")
	(run-sed canon-path "s/<([^x>]*)x([^x>]*)x([^x>]*)x([^x>]*)x([^x>]*)>/<\1 × \2 × \3 × \4 × \5>/g")
	(run-sed canon-path "s/<([^x>]*)x([^x>]*)x([^x>]*)x([^x>]*)x([^x>]*)x([^x>]*)>/<\1 × \2 × \3 × \4 × \5 × \6>/g")
	(run-sed canon-path "s/^#.*//") ;; remove attribute aliases
	;; (run-sed canon-path "s/e\+//g") ;; remove floating point 0e+0 with 0e0
	(run-sed canon-path "s-//.*--g") ;; remove comments
	(run-sed canon-path "s/<([^x>]*)x([^x>]*)>/<\1 × \2>/g")
	(run-sed canon-path "s/<([^x>]*)x([^x>]*)>/<\1 × \2>/g")
	(setf (mlir-file-part-canon-contents part) (str:from-file canon-path))))))


(defstruct stat nsucc nfail)


;;  print statistics over data, gathered by running stat-successp on each
;; element of data-list
(defun print-stats (data-list stat-successp)
  (let ((stats (make-hash-table :test 'equal)))
    (loop for part in data-list do
      (let ((s (ensure-gethash (mlir-file-part-category part)
			       stats
			       (make-stat :nsucc 0 :nfail 0))))
	(if (funcall stat-successp part)
	    (incf (stat-nsucc s))
	    (incf (stat-nfail s))
	    )))
    (format t "compile stats: |~d|~%" stats)
    (loop for k being the hash-keys of stats using (hash-value v) do
      (format t "~a => (SUCC ~a | FAIL ~a)/~a~%"
	      k
	      (stat-nsucc v)
	      (stat-nfail v)
	      (+ (stat-nsucc v) (stat-nfail v))))))

(defun print-compile-stats ()
  (print-stats *canon-mlir-parts* #'mlir-file-part-compile-successp))

;;  TODO: canonicalize output of running 
(defun print-run-stats ()
  (print-stats *canon-mlir-parts* #'mlir-file-part-run-canon-successp))


(defun main ()
  (setf lparallel:*kernel* (lparallel:make-kernel 32))
  (loop for (category path-root) in *tests* do
    (format t "collecting category |~d| at  path |~d|~%" category path-root)
    (let* ((paths (directory path-root)))
      (format t "..found |~d| files~%" (length paths))
      (loop for path in paths do
	;; ignore invalid test files because they're literally invalid
	(unless (str:contains? "invalid" (pathname-name path))
	  (format t "....found path |~d|~%" path)
	  (setf *raw-mlir-parts* (append *raw-mlir-parts* (read-mlir-file category path)))))))

  (format t "found total |~d| parts |~d| ~%"
	  (length *raw-mlir-parts*) *raw-mlir-parts*)
  ;; try to canonicalize
  (lparallel:pmap nil #'canonicalize-mlir-part *raw-mlir-parts*)
  ;; TODO: reindex the guid based on the new array length after filtering.
  ;; create lean files
  (lparallel:pmap nil #'make-and-run-lean-file *canon-mlir-parts*)
  (print-compile-stats)
  (print-run-stats)
  (uiop:with-output-file (outf #P"./test.csv" :if-exists :supersede)
    (cl-csv:write-csv (cons *csv-preamble* (mapcar #'mlir-part-to-csv-row *canon-mlir-parts*)) :stream outf))
  ;; (uiop:with-output-file (outf #P"./test.json" :if-exists :supersede)
  ;;   (shasht:write-json (list (cons :stats *run-stats*)) outf))
  )
(main)
