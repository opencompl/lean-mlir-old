import MLIR.Tests.TestLib
import MLIR.Tests.SemanticsTests
import MLIR.Tests.DominanceTests

open TestLib

namespace AllTests

def testSuite : TestSuite := [
  SemanticsTests.testGroup,
  DominanceTests.testGroup
]

end AllTests