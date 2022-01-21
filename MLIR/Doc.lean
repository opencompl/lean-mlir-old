import Init.Data.String
import Init.Data.String.Basic
import Init.Data.Char.Basic
import Init.System.IO
import Lean.Parser
import Lean.Parser.Extra
import Init.System.Platform
import Init.Data.String.Basic
import Init.Data.Repr
import Init.Data.ToString.Basic

-- | TODO: Consider adopting flutter rendering model: 
-- linear time, flexbox, one walk up and one walk down tree.
-- https://www.youtube.com/watch?v=UUfXWzp0-DU

namespace MLIR.Doc

inductive Doc : Type where
  | Concat : Doc -> Doc -> Doc
  | Nest : Doc -> Doc
  | VGroup : List Doc -> Doc
  | Text: String -> Doc


class Pretty (a : Type) where
  doc : a -> Doc

open Pretty

def vgroup [Pretty a] (as: List a): Doc :=
  Doc.VGroup (as.map doc)

def nest_vgroup [Pretty a] (as: List a): Doc :=
  Doc.Nest (vgroup as)


instance : Pretty Doc where
  doc (d: Doc) := d

instance : Pretty String where
  doc := Doc.Text

instance : Pretty Int where
  doc := Doc.Text ∘ toString

instance : Pretty Char where
  doc := Doc.Text ∘ toString

instance : Inhabited Doc where
  default := Doc.Text ""


instance : Coe String Doc where
  coe := Doc.Text

instance : Append Doc where 
  append := Doc.Concat

def doc_dbl_quot : Doc :=  doc '"'

def doc_surround_dbl_quot [Pretty a] (v: a): Doc := 
    doc_dbl_quot ++ doc v ++ doc_dbl_quot
  

def doc_concat (ds: List Doc): Doc := ds.foldl Doc.Concat (Doc.Text "") 

partial def intercalate_doc_rec_ [Pretty d] (ds: List d) (i: Doc): Doc :=
  match ds with
  | [] => Doc.Text ""
  | (d::ds) => i ++ (doc d) ++ intercalate_doc_rec_ ds i

partial def  intercalate_doc [Pretty d] (ds: List d) (i: Doc): Doc := match ds with
 | [] => Doc.Text ""
 | [d] => doc d
 | (d::ds) => (doc d) ++ intercalate_doc_rec_ ds i

 
partial def vintercalate_doc_rec_ [Pretty d] (ds: List d) (i: String): List Doc :=
  match ds with
  | [] => [Doc.Text ""]
  | (d::ds) => (i ++ (doc d)) :: vintercalate_doc_rec_ ds i

partial def  vintercalate_doc [Pretty d] (ds: List d) (i: String): Doc := match ds with
 | [] => Doc.Text ""
 | [d] => doc d
 | (d::ds) => Doc.VGroup $ (doc d)::vintercalate_doc_rec_ ds i
             


partial def layout 
  (d: Doc)
  (indent: Int) -- indent
  (width: Int) -- width
  (leftover: Int) -- characters left
  (newline: Bool) -- create newline?
  : String :=
  match d with
    | (Doc.Text s)  => (if newline then "\n".pushn ' ' indent.toNat else "") ++ s
    | (Doc.Concat d1 d2) =>
         let s := layout d1 indent width leftover newline
         s ++ layout d2 indent width (leftover - (s.length + 1)) false
    | (Doc.Nest d) => layout d (indent+1) width leftover newline
    | (Doc.VGroup ds) => 
       let ssInline := layout (doc_concat ds) indent width leftover newline 
       if false then ssInline -- ssInline.length <= leftover then ssInline
       else  
         let width' := width - indent
         -- TODO: don't make 
         String.join (ds.map (fun d => layout d indent width width True))


def layout80col (d: Doc) : String := layout d 0 80 0 false

instance : Coe Doc String where
   coe := layout80col

end MLIR.Doc
