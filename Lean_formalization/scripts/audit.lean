import ChannelFormalization
import Lean.Util.CollectAxioms

open Lean Elab Command

/- Audit transitive kernel-axiom dependencies of every project theorem, including
private theorems. Run with `./lakew env lean scripts/audit.lean` after `./lakew build`. -/
run_cmd do
  let env ← getEnv
  let theoremNames := env.constants.toList.filterMap fun (name, info) =>
    let userName := (privateToUserName? name).getD name
    if (`ChannelFormalization).isPrefixOf userName && info.isTheorem then some name else none
  let theoremNames := theoremNames.toArray.qsort Name.quickLt
  if theoremNames.isEmpty then
    throwError "Axiom audit found no ChannelFormalization theorems; check the aggregate imports."
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut used : Array Name := #[]
  let mut failures : Array MessageData := #[]
  for name in theoremNames do
    let dependencies ← collectAxioms name
    for dependency in dependencies do
      if !used.contains dependency then
        used := used.push dependency
    let forbidden := dependencies.filter fun dependency => !allowed.contains dependency
    if !forbidden.isEmpty then
      failures := failures.push m!"{name}: {forbidden.toList}"
  if !failures.isEmpty then
    throwError m!"Transitive axiom audit failed:\n{MessageData.joinSep failures.toList "\n"}"
  logInfo m!"Transitive axiom audit passed for {theoremNames.size} ChannelFormalization theorems."
  logInfo m!"Axioms used: {(used.qsort Name.quickLt).toList}"
