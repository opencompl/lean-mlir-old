import MLIR.AST

-- https://mlir.llvm.org/docs/Dialects/EmitC/
-- https://www.lysator.liu.se/c/ANSI-C-grammar-y.html

declare_syntax_cat primary_expression
declare_syntax_cat postfix_expression
declare_syntax_cat argument_expression_list
declare_syntax_cat unary_expression
declare_syntax_cat unary_operator
declare_syntax_cat cast_expression
declare_syntax_cat multiplicative_expression
declare_syntax_cat additive_expression
declare_syntax_cat shift_expression
declare_syntax_cat relational_expression
declare_syntax_cat equality_expression
declare_syntax_cat and_expression
declare_syntax_cat exclusive_or_expression
declare_syntax_cat inclusive_or_expression
declare_syntax_cat logical_and_expression
declare_syntax_cat logical_or_expression
declare_syntax_cat conditional_expression
declare_syntax_cat assignment_expression
declare_syntax_cat assignment_operator
declare_syntax_cat expression
declare_syntax_cat constant_expression
declare_syntax_cat declaration
declare_syntax_cat declaration_specifiers
declare_syntax_cat init_declarator_list
declare_syntax_cat init_declarator
declare_syntax_cat storage_class_specifier
declare_syntax_cat type_specifier
declare_syntax_cat struct_or_union_specifier
declare_syntax_cat struct_or_union
declare_syntax_cat struct_declaration_list
declare_syntax_cat struct_declaration
declare_syntax_cat specifier_qualifier_list
declare_syntax_cat struct_declarator_list
declare_syntax_cat struct_declarator
declare_syntax_cat enum_specifier
declare_syntax_cat enumerator_list
declare_syntax_cat enumerator
declare_syntax_cat type_qualifier
declare_syntax_cat declarator
declare_syntax_cat direct_declarator
declare_syntax_cat pointer     
declare_syntax_cat type_qualifier_list
declare_syntax_cat parameter_type_list
declare_syntax_cat parameter_list
declare_syntax_cat parameter_declaration
declare_syntax_cat identifier_list
declare_syntax_cat type_name
declare_syntax_cat abstract_declarator
declare_syntax_cat direct_abstract_declarator
declare_syntax_cat initializer
declare_syntax_cat initializer_list
declare_syntax_cat statement
declare_syntax_cat labeled_statement
declare_syntax_cat compound_statement
declare_syntax_cat declaration_list
declare_syntax_cat statement_list
declare_syntax_cat expression_statement
declare_syntax_cat selection_statement
declare_syntax_cat iteration_statement
declare_syntax_cat jump_statement
declare_syntax_cat translation_unit
declare_syntax_cat external_declaration
declare_syntax_cat function_definition


-- primary expression
syntax ident : primary_expression
syntax num: primary_expression
syntax str: primary_expression
syntax "(" expression ")" : primary_expression

-- postfix expression
syntax primary_expression : postfix_expression
syntax postfix_expression "[" expression "]" : postfix_expression
syntax postfix_expression "(" ")" : postfix_expression
syntax postfix_expression "(" argument_expression_list ")" : postfix_expression
syntax postfix_expression "." ident : postfix_expression
syntax postfix_expression "->" ident : postfix_expression
syntax postfix_expression "++" : postfix_expression
syntax postfix_expression "--" : postfix_expression


-- argument_expression_list
syntax assignment_expression : argument_expression_list
syntax argument_expression_list "," assignment_expression : argument_expression_list


-- unary expression
syntax postfix_expression : unary_expression
syntax "++" unary_expression : unary_expression
syntax "--" unary_expression : unary_expression
syntax unary_operator cast_expression : unary_expression
syntax "sizeof" unary_expression : unary_expression
syntax "sizeof" "(" type_name ")" : unary_expression

-- unary operator
syntax "&" : unary_operator
syntax "*": unary_operator
syntax "+": unary_operator
syntax "-": unary_operator
syntax "~": unary_operator
syntax "!" : unary_operator

-- cast expression
syntax unary_expression: cast_expression 
syntax "(" type_name ")" : cast_expression

-- multiplicative expression
syntax cast_expression : multiplicative_expression
syntax multiplicative_expression "*" cast_expression : multiplicative_expression
syntax multiplicative_expression "/" cast_expression : multiplicative_expression
syntax multiplicative_expression "%" cast_expression : multiplicative_expression

-- additive expression
syntax multiplicative_expression : additive_expression
syntax additive_expression "+" multiplicative_expression : additive_expression
syntax additive_expression "-" multiplicative_expression : additive_expression

-- shift_expression
syntax additive_expression : shift_expression
syntax shift_expression "<<" additive_expression : shift_expression
syntax shift_expression ">>" additive_expression : shift_expression

-- relational_expression
syntax shift_expression : relational_expression
syntax relational_expression "<" shift_expression : relational_expression
syntax relational_expression ">" shift_expression : relational_expression
syntax relational_expression "<=" shift_expression :relational_expression                  
syntax relational_expression ">=" shift_expression : relational_expression

-- equality_expression
syntax  relational_expression : equality_expression
syntax  equality_expression "==" relational_expression : equality_expression
syntax  equality_expression "!=" relational_expression : equality_expression

-- and_expression
syntax equality_expression : and_expression
syntax and_expression "&" equality_expression : and_expression

-- exclusive_or_expression 
syntax and_expression : exclusive_or_expression 
syntax exclusive_or_expression "^" and_expression : exclusive_or_expression 

-- inclusive_or_expression
syntax exclusive_or_expression : inclusive_or_expression
syntax inclusive_or_expression "|" exclusive_or_expression : inclusive_or_expression 

-- logical_and_expression
syntax inclusive_or_expression : logical_and_expression
syntax logical_and_expression "&&" inclusive_or_expression : logical_and_expression

-- logical_or_expression
syntax logical_and_expression : logical_or_expression
syntax logical_or_expression "||" logical_and_expression : logical_or_expression


-- conditional_expression
syntax logical_or_expression : conditional_expression
syntax logical_or_expression "?" expression ":" conditional_expression : conditional_expression

-- assignment_expression
syntax conditional_expression : assignment_expression
syntax unary_expression assignment_operator assignment_expression : assignment_expression

-- assignment_operator
syntax "=" : assignment_operator
syntax "*=" : assignment_operator
syntax "/=" : assignment_operator
syntax "%=" : assignment_operator
syntax "+=" : assignment_operator
syntax "-=" : assignment_operator
syntax "<<=" : assignment_operator
syntax ">>=" : assignment_operator
syntax "&=" : assignment_operator
syntax "^=" : assignment_operator
syntax "|=" : assignment_operator

-- expression
syntax assignment_expression : expression
syntax expression "," assignment_expression : expression

-- constant_expression
syntax conditional_expression : constant_expression

-- syntax declaration
syntax declaration_specifiers ";" : declaration
syntax declaration_specifiers init_declarator_list ";" : declaration

-- declaration_specifiers
syntax storage_class_specifier : declaration_specifiers
syntax storage_class_specifier declaration_specifiers : declaration_specifiers
syntax type_specifier: declaration_specifiers
syntax type_specifier declaration_specifiers: declaration_specifiers
syntax type_qualifier: declaration_specifiers
syntax type_qualifier declaration_specifiers: declaration_specifiers


-- init_declarator_list
syntax init_declarator : init_declarator_list
syntax init_declarator_list "," init_declarator : init_declarator_list

-- init_declarator
syntax declarator : init_declarator
syntax declarator "=" initializer : init_declarator

-- storage_class_specifier
syntax "typedef" : storage_class_specifier
syntax "extern": storage_class_specifier
syntax "static": storage_class_specifier
syntax "auto": storage_class_specifier
syntax "register": storage_class_specifier

-- type_specifier
syntax "void" : type_specifier
syntax "char": type_specifier
syntax "short": type_specifier
syntax "int": type_specifier
syntax "long": type_specifier
syntax "float": type_specifier
syntax "double": type_specifier
syntax "signed" : type_specifier
syntax "unsigned": type_specifier
syntax struct_or_union_specifier: type_specifier
syntax enum_specifier: type_specifier
syntax "typename" : type_specifier


-- struct_or_union_specifier
syntax struct_or_union ident "{" struct_declaration_list "}" : struct_or_union_specifier
syntax struct_or_union "{" struct_declaration_list "}" : struct_or_union_specifier
syntax struct_or_union ident : struct_or_union_specifier

-- struct_or_union
syntax "struct" : struct_or_union
syntax "union" : struct_or_union

-- struct_declaration_list
syntax struct_declaration : struct_declaration_list
syntax struct_declaration_list struct_declaration : struct_declaration_list

-- struct_declaration
syntax specifier_qualifier_list struct_declarator_list ";" : struct_declaration

-- specifier_qualifier_list
syntax type_specifier specifier_qualifier_list : specifier_qualifier_list
syntax type_specifier : specifier_qualifier_list
syntax type_qualifier specifier_qualifier_list : specifier_qualifier_list
syntax type_qualifier : specifier_qualifier_list

-- struct_declarator_list
syntax struct_declarator : struct_declarator_list
syntax struct_declarator_list "," struct_declarator : struct_declarator_list

-- struct_declarator
syntax declarator : struct_declarator
syntax ":" constant_expression : struct_declarator
syntax declarator ":" constant_expression : struct_declarator

-- enum_specifier
syntax "enum" "{" enumerator_list "}"  : enum_specifier
syntax "enum" ident "{" enumerator_list "}" : enum_specifier
syntax "enum" ident : enum_specifier

-- enumerator_list
syntax enumerator : enumerator_list
syntax enumerator_list "," enumerator : enumerator_list


-- enumerator
syntax ident : enumerator
syntax ident "=" constant_expression : enumerator

-- type_qualifier
syntax "const" : type_qualifier
syntax "volatile" :type_qualifier

-- declarator
syntax pointer direct_declarator :declarator 
syntax direct_declarator : declarator

-- direct_declarator
syntax ident : direct_declarator
syntax "(" declarator ")" : direct_declarator
syntax direct_declarator "[" constant_expression "]" : direct_declarator
syntax direct_declarator "[" "]": direct_declarator
syntax direct_declarator "(" parameter_type_list ")": direct_declarator
syntax direct_declarator "(" identifier_list ")": direct_declarator
syntax direct_declarator "(" ")": direct_declarator

-- pointer
syntax "*" : pointer
syntax "*" type_qualifier_list : pointer
syntax "*" pointer : pointer
syntax "*" type_qualifier_list pointer : pointer


-- type_qualifier_list
syntax type_qualifier : type_qualifier_list
syntax type_qualifier_list type_qualifier : type_qualifier_list

-- parameter_type_list
syntax parameter_list : parameter_type_list
syntax parameter_list "," "..." : parameter_type_list

-- parameter_list
syntax parameter_declaration : parameter_list
syntax parameter_list "," parameter_declaration : parameter_list

-- parameter_declaration
syntax declaration_specifiers declarator : parameter_declaration
syntax declaration_specifiers abstract_declarator : parameter_declaration
syntax declaration_specifiers : parameter_declaration

-- identifier_list
syntax ident : identifier_list
syntax identifier_list "," ident : identifier_list

-- type_name
syntax specifier_qualifier_list : type_name
syntax specifier_qualifier_list abstract_declarator : type_name


-- abstract_declarator
syntax pointer : abstract_declarator
syntax direct_abstract_declarator : abstract_declarator
syntax pointer direct_abstract_declarator : abstract_declarator


-- direct_abstract_declarator
syntax "(" abstract_declarator ")" : direct_abstract_declarator
syntax "[" "]" : direct_abstract_declarator
syntax "[" constant_expression "]" : direct_abstract_declarator
syntax direct_abstract_declarator "[" "]" : direct_abstract_declarator
syntax direct_abstract_declarator "[" constant_expression "]" : direct_abstract_declarator
syntax "(" ")" : direct_abstract_declarator
syntax "(" parameter_type_list ")" : direct_abstract_declarator
syntax direct_abstract_declarator "(" ")" : direct_abstract_declarator
syntax direct_abstract_declarator "(" parameter_type_list ")" : direct_abstract_declarator

-- initializer
syntax assignment_expression : initializer
syntax "{" initializer_list "}" : initializer
syntax "{" initializer_list "," "}" : initializer

-- initializer_list
syntax initializer : initializer_list
syntax initializer_list "," initializer : initializer_list

-- statement
syntax labeled_statement : statement
syntax compound_statement : statement
syntax expression_statement : statement
syntax selection_statement : statement
syntax iteration_statement : statement
syntax jump_statement : statement

-- labeled_statement
syntax ident "syntax" statement : labeled_statement
syntax "case" constant_expression ":" statement : labeled_statement
syntax "default" ":" statement : labeled_statement

-- compound_statement
syntax "{" "}" : compound_statement
syntax "{" statement_list "}" : compound_statement
syntax "{" declaration_list "}" : compound_statement
syntax "{" declaration_list statement_list "}" : compound_statement

-- declaration_list
syntax declaration : declaration_list
syntax declaration_list declaration : declaration_list

-- statement_list
syntax statement : statement_list
syntax statement_list statement : statement_list

-- expression_statement
syntax ";" : expression_statement
syntax expression ";" : expression_statement

-- selection_statement
syntax "if" "(" expression ")" statement : selection_statement
syntax "if" "(" expression ")" statement "else" statement : selection_statement
syntax "switch" "(" expression ")" statement : selection_statement

-- iteration_statement
syntax "while" "(" expression ")" statement : iteration_statement
syntax "do" statement "while" "(" expression ")" ";" : iteration_statement
syntax "for" "(" expression_statement expression_statement ")" statement : iteration_statement
syntax "for" "(" expression_statement expression_statement expression ")" statement : iteration_statement

-- jump_statement
syntax "goto" ident ";" : jump_statement 
syntax "continue" ";" : jump_statement 
syntax "break" ";" : jump_statement 
syntax "return" ";" : jump_statement 
syntax "return" expression ";" : jump_statement 

-- translation_unit
syntax external_declaration : translation_unit
syntax translation_unit external_declaration : translation_unit

-- external_declaration
syntax function_definition : external_declaration
syntax declaration : external_declaration


-- function_definition
syntax declaration_specifiers declarator declaration_list compound_statement : function_definition
syntax declaration_specifiers declarator compound_statement : function_definition
syntax declarator declaration_list compound_statement : function_definition
syntax declarator compound_statement : function_definition