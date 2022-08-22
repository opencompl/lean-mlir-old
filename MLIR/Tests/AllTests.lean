import MLIR.Tests.TestLib
import MLIR.Tests.SemanticsTests

open TestLib

namespace AllTests

def testSuite : TestSuite := [
  SemanticsTests.testGroup
]

end AllTests