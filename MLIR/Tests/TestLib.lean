
namespace TestLib

abbrev TestCase := String × Except String Unit
abbrev TestGroup := String × List TestCase
abbrev TestSuite := List TestGroup

/- Execute the tests, and return the number of succeeded and failed tests. -/
def runTestList (tests: List TestCase) : IO (Nat × Nat) := do
  let mut failed := 0
  let mut succeeded := 0
  for (name, test) in tests do
    match test with
    | .ok () => IO.println s!"PASS {name}"; succeeded := succeeded + 1
    | .error e => IO.println s!"FAIL {name}: {e}"; failed := failed + 1
  return (succeeded, failed)

/- Run a test group, and return the number of succeeded and failed tests. -/
def runTestGroup (group: TestGroup) : IO (Nat × Nat) := do
  IO.println s!"Running {group.fst} tests..."
  runTestList group.snd

/- Run all test groups, and return false if a test failed -/
def runTestSuite (groups: TestSuite) : IO Bool := do
  let mut failed := 0
  let mut succeeded := 0
  for group in groups do
    let (s, f) ← runTestGroup group
    failed := failed + f
    succeeded := succeeded + s
  IO.println s!"PASSED tests: {succeeded}"
  IO.println s!"FAILED tests: {failed}"
  return failed = 0

end TestLib