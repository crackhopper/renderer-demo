## ADDED Requirements

### Requirement: GlobalStringTable provides unique ID for each string
`GlobalStringTable` SHALL maintain a global mapping from string to `uint32_t` ID. For any given string, the returned ID MUST be the same across all calls, regardless of calling thread or time.

#### Scenario: First registration of a string
- **WHEN** `getOrCreateID("u_BaseColor")` is called for the first time
- **THEN** a new unique `uint32_t` ID is returned and stored internally

#### Scenario: Repeated query of same string
- **WHEN** `getOrCreateID("u_BaseColor")` is called again after initial registration
- **THEN** the same ID as the first call is returned

#### Scenario: Different strings get different IDs
- **WHEN** `getOrCreateID("u_BaseColor")` and `getOrCreateID("u_Time")` are called
- **THEN** the two returned IDs MUST be different

### Requirement: GlobalStringTable supports reverse lookup
`GlobalStringTable` SHALL support looking up the original string from an ID, for debugging and logging purposes.

#### Scenario: Reverse lookup of registered ID
- **WHEN** `getName(id)` is called with an ID previously returned by `getOrCreateID`
- **THEN** the original string is returned

#### Scenario: Reverse lookup of unknown ID
- **WHEN** `getName(id)` is called with an ID that was never assigned
- **THEN** a fallback string (e.g. `"UNKNOWN_PROPERTY"`) is returned

### Requirement: GlobalStringTable is thread-safe
All operations on `GlobalStringTable` SHALL be safe to call from multiple threads concurrently without data races.

#### Scenario: Concurrent registration from multiple threads
- **WHEN** multiple threads call `getOrCreateID` with different strings simultaneously
- **THEN** each thread receives a correct, unique ID without data corruption

#### Scenario: Concurrent read and write
- **WHEN** one thread calls `getOrCreateID` (write) while another calls `getName` (read)
- **THEN** both operations complete correctly without data races

### Requirement: StringID wraps uint32_t with string construction
`StringID` SHALL be a struct wrapping a `uint32_t` ID. It MUST support implicit construction from `const char*` and `const std::string&` via `GlobalStringTable`. Construction from `uint32_t` MUST be `explicit`.

#### Scenario: Implicit construction from string literal
- **WHEN** a function accepting `StringID` is called with `"u_Time"`
- **THEN** a `StringID` is implicitly constructed with the ID for `"u_Time"` from `GlobalStringTable`

#### Scenario: Equality comparison
- **WHEN** two `StringID` values constructed from the same string are compared with `==`
- **THEN** the comparison returns `true`

#### Scenario: Use as unordered_map key
- **WHEN** `StringID` is used as a key in `std::unordered_map<StringID, T, StringID::Hash>`
- **THEN** it works correctly with zero hash collisions (since IDs are unique integers)

### Requirement: MakeStringID convenience function
A free function `MakeStringID(const std::string&)` SHALL be provided as a convenience wrapper around `GlobalStringTable::get().getOrCreateID()`.

#### Scenario: MakeStringID returns same ID as StringID constructor
- **WHEN** `MakeStringID("u_Color")` is called and separately `StringID("u_Color")` is constructed
- **THEN** both have the same underlying `uint32_t` ID

### Requirement: MaterialTemplate uses StringID for binding cache
`MaterialTemplate::m_bindingCache` SHALL use `StringID` as its key type instead of `std::string`.

#### Scenario: Building binding cache from shader reflection
- **WHEN** `buildBindingCache()` is called
- **THEN** each shader binding name is converted to a `StringID` and stored in the cache

#### Scenario: Finding binding by StringID
- **WHEN** `findBinding(StringID("u_BaseColor"))` is called
- **THEN** the corresponding `ShaderResourceBinding` is returned if it exists

### Requirement: MaterialInstance uses StringID for property storage
`MaterialInstance` SHALL use `StringID` as the key type for all property maps (`m_vec4s`, `m_floats`, `m_textures`). The `setVec4`, `setFloat`, and `setTexture` methods SHALL accept `StringID` as their first parameter.

#### Scenario: Setting a float property by string
- **WHEN** `mat->setFloat("u_Time", 1.0f)` is called
- **THEN** `"u_Time"` is implicitly converted to `StringID` and the value is stored under that ID

### Requirement: TypeTag enumerates structured StringID categories
`GlobalStringTable` SHALL provide a `TypeTag` enum (`uint8_t` backing) that distinguishes leaf strings from each category of structured StringID. The enum MUST include at minimum `String`, `ShaderProgram`, `RenderState`, `VertexLayoutItem`, `VertexLayout`, `MeshRender`, `Skeleton`, `RenderPassEntry`, `MaterialRender`, `ObjectRender`, and `PipelineKey`. Leaf strings interned via `Intern` or `getOrCreateID` MUST be tagged as `TypeTag::String`. Topology and similar small leaves SHALL remain plain strings and MUST NOT be represented as a separate `TypeTag`.

#### Scenario: Leaf string has String tag
- **WHEN** `Intern("tri")` is called and the result is passed to `decompose`
- **THEN** the returned `Decomposed::tag` equals `TypeTag::String` and `fields` is empty

#### Scenario: Structured compose tag is preserved
- **WHEN** `compose(TypeTag::PipelineKey, {a, b})` is called and the result is passed to `decompose`
- **THEN** the returned `Decomposed::tag` equals `TypeTag::PipelineKey`

### Requirement: Intern provides an explicit leaf entry point
`GlobalStringTable::Intern(std::string_view)` SHALL return a `StringID` equivalent to `StringID(std::string(sv))` and SHALL be semantically identical to `getOrCreateID` followed by `StringID` construction. This method exists to give structured code paths an unambiguous leaf-intern call site without relying on implicit `StringID` construction.

#### Scenario: Intern deduplicates leaves
- **WHEN** `Intern("foo")` is called twice
- **THEN** both calls return `StringID` values with the same underlying `uint32_t` id

#### Scenario: Intern agrees with implicit StringID
- **WHEN** `Intern("foo")` and `StringID("foo")` are constructed in any order
- **THEN** both have the same underlying `uint32_t` id

### Requirement: compose produces a deduplicated structured StringID
`GlobalStringTable::compose(TypeTag tag, std::span<const StringID> fields)` SHALL return a `StringID` that is uniquely determined by the pair `(tag, fields)` including field order. Two calls with the same tag and the same field sequence (by `uint32_t` id) MUST return the same `StringID`. The internal canonicalization key SHALL be a string of the form `"{TagName}({field0.id},{field1.id},...)"` that is interned via the existing `m_stringToId` path, ensuring no additional hash-collision handling is required. The compose table SHALL also record the `(tag, fields)` metadata for later `decompose`.

#### Scenario: Compose deduplicates same inputs
- **WHEN** `compose(PipelineKey, {a, b})` is called twice with the same `a` and `b`
- **THEN** both calls return `StringID` values with the same underlying id

#### Scenario: Compose is order sensitive
- **WHEN** `compose(PipelineKey, {a, b})` and `compose(PipelineKey, {b, a})` are called with distinct `a != b`
- **THEN** the two returned `StringID` values have different underlying ids

#### Scenario: Compose with different tags produces different ids
- **WHEN** `compose(PipelineKey, {a, b})` and `compose(ObjectRender, {a, b})` are called
- **THEN** the two returned `StringID` values have different underlying ids

#### Scenario: Compose with empty fields
- **WHEN** `compose(MeshRender, {})` is called
- **THEN** it returns a valid `StringID` and `decompose` on it returns tag `MeshRender` with an empty `fields` vector

### Requirement: decompose reverses compose and identifies leaves
`GlobalStringTable::decompose(StringID id)` SHALL return `std::optional<Decomposed>` where `Decomposed { TypeTag tag; std::vector<StringID> fields; }`. If `id` was produced by a prior `compose` call, the returned value MUST contain the original `(tag, fields)`. If `id` is a valid leaf (produced by `Intern` or `getOrCreateID`), the returned value MUST have `tag == TypeTag::String` and empty `fields`. If `id` is not valid in this table, the function MUST return `std::nullopt`.

#### Scenario: Round-trip of a compose call
- **WHEN** `decompose(compose(PipelineKey, {a, b}))` is evaluated
- **THEN** the result is a `Decomposed` with `tag == PipelineKey` and `fields == {a, b}`

#### Scenario: Decompose of a leaf
- **WHEN** `decompose(Intern("foo"))` is evaluated
- **THEN** the result is a `Decomposed` with `tag == TypeTag::String` and empty `fields`

#### Scenario: Decompose of an invalid id
- **WHEN** `decompose(StringID{0xDEADBEEF})` is evaluated for an id never produced by this table
- **THEN** the result is `std::nullopt`

### Requirement: toDebugString recursively renders structured ids
`GlobalStringTable::toDebugString(StringID id)` SHALL return a human-readable string. A leaf returns its original string. A structured id returns `"<TagName>(<child1>, <child2>, ...)"` where each child is recursively rendered via `toDebugString`. The implementation MUST guard against pathological cycles with a fixed maximum recursion depth (at least 16); on exceeding the depth it MUST return the literal `"<...>"` for the offending subtree rather than recurse further.

#### Scenario: Leaf rendering
- **WHEN** `toDebugString(Intern("foo"))` is called
- **THEN** it returns `"foo"`

#### Scenario: Nested structured rendering
- **WHEN** `toDebugString(compose(PipelineKey, {Intern("foo"), compose(VertexLayout, {Intern("pos")})}))` is called
- **THEN** it returns `"PipelineKey(foo, VertexLayout(pos))"`

#### Scenario: Depth guard
- **WHEN** `toDebugString` is called on an id whose recursive expansion exceeds the configured depth limit
- **THEN** the returned string contains `"<...>"` in place of the too-deep subtree and the call does not overflow the stack

### Requirement: Structured interning operations are thread-safe
All `Intern`, `compose`, `decompose`, and `toDebugString` operations SHALL be safe to call from multiple threads concurrently. They MUST share the existing `shared_mutex` protecting `m_stringToId` / `m_idToString`, with the new `m_composedEntries` map protected under the same lock (shared reads, unique writes).

#### Scenario: Concurrent compose of the same inputs
- **WHEN** N threads simultaneously call `compose(PipelineKey, {a, b})` with identical arguments
- **THEN** every thread receives a `StringID` with the same underlying id and no data race occurs

#### Scenario: Concurrent compose and decompose
- **WHEN** one thread calls `compose` while another calls `decompose` on a previously composed id
- **THEN** both calls complete correctly without data races
