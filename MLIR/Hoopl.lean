-- An implementation of the Hoopl algorithm for MLIR.
-- see: https://github.com/bollu/mlir-hoopl-rete

-- {}
-- * x = 1
-- ANALYZE: { x = 1 }
-- * y = 2 
-- ANALYZE: {x = 1, y = 2 }
-- * P1: z = x + y
-- REWRITE: z = 3  (evaluate + in the context where x, y was known)
-- * P2: z = 3 
-- ANALYZE: {x = 1, y = 2, z = 3 } (analyze CONSTANT assignment)
-- ANALYZE { x = 1, y = 2, z = 3}
-- * w = z + 1
-- ANALYZE {x = 1, y = 2, z = 3, w = 4 }






