constant foo : Unit
constant bar : Unit

unsafe def fooImpl : Unit := bar
unsafe def barImpl : Unit := foo

attribute [implementedBy fooImpl] foo
attribute [implementedBy barImpl] bar
