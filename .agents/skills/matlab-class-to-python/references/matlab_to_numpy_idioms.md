# MATLAB to NumPy / SciPy Idioms Reference (Horace & Herbert)

This reference provides exact, numerically equivalent Python/NumPy/SciPy translations for MATLAB array, indexing, binning, and set operations commonly used in Horace and Herbert.

---

## 1. Indexing, Reshaping, and Linear Coordinates

MATLAB uses **1-based, column-major (Fortran)** indexing. NumPy uses **0-based, row-major (C)** indexing by default.

| MATLAB Operation | NumPy Equivalent | Critical Parity Notes |
|---|---|---|
| `A(i, j)` | `A[i - 1, j - 1]` | Subtract 1 from 1-based MATLAB indices. |
| `A(1:k)` | `A[:k]` | MATLAB `1:k` is inclusive ($k$ elements); Python `0:k` is half-open ($k$ elements). |
| `A(i:step:j)` | `A[i-1 : j : step]` (if `step > 0`) | Ensure stop bound includes `j` when converting inclusive ranges. |
| `A(:)` | `A.reshape(-1, order="F")` or `A.ravel(order="F")` | MATLAB `A(:)` always column-stacks in Fortran order. |
| `reshape(A, m, n)` | `np.reshape(A, (m, n), order="F")` | Omit `order="F"` only if the array is 1D. |
| `A(linear_idx)` | `A.ravel(order="F")[linear_idx - 1]` | Linear indexing into a 2D/3D array in MATLAB traverses down columns first. |
| `ind = sub2ind(sz, r, c)` | `np.ravel_multi_index((r - 1, c - 1), dims=sz, order="F")` | Returns 0-based linear index (add 1 only if serializing 1-based indices). |
| `[r, c] = ind2sub(sz, ind)` | `np.unravel_index(ind - 1, shape=sz, order="F")` | Returns 0-based coordinate tuple `(r0, c0)`. |
| `find(cond)` | `np.flatnonzero(cond)` (1D) or `np.nonzero(cond)` | For 2D linear `find(A > 0)` matching MATLAB column-major order: `np.flatnonzero(np.asarray(A > 0).ravel(order="F"))`. |
| `find(cond, k, 'first')` | `np.flatnonzero(cond)[:k]` | First `k` matching indices. |
| `find(cond, k, 'last')` | `np.flatnonzero(cond)[-k:]` | Last `k` matching indices. |

---

## 2. Set Operations (`unique`, `ismember`, `sortrows`)

### `unique` (Sorted vs. `'stable'` and `'rows'`)
* **Standard sorted `[C, ia, ic] = unique(A)`**:
  ```python
  # Note: MATLAB flattens A column-wise unless 'rows' is specified
  C, ia, ic = np.unique(A.ravel(order="F"), return_index=True, return_inverse=True)
  ```
* **Order-preserving `[C, ia, ic] = unique(A, 'stable')`**:
  NumPy's `np.unique` always sorts output `C`. To preserve first-appearance order (`'stable'`):
  ```python
  flat = np.asarray(A).ravel(order="F")
  u_vals, u_first_idx, u_inv = np.unique(
      flat, return_index=True, return_inverse=True
  )
  order = np.argsort(u_first_idx)
  C = u_vals[order]
  ia = u_first_idx[order]
  # Map inverse indices to the new 'stable' order
  inv_map = np.empty_like(order)
  inv_map[order] = np.arange(len(order))
  ic = inv_map[u_inv]
  ```
* **Row-wise `[C, ia, ic] = unique(A, 'rows')`**:
  ```python
  C, ia, ic = np.unique(A, axis=0, return_index=True, return_inverse=True)
  ```

### `ismember`
* **`tf = ismember(A, B)`**:
  ```python
  tf = np.isin(A, B)
  ```
* **`[tf, loc] = ismember(A, B)`** (where `loc` is the first matching index in `B`, or `-1` / `0` if not found):
  ```python
  def ismember(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Return (tf, loc) where loc contains 0-based index in b (or -1 if not present)."""
      a_arr = np.asarray(a)
      b_arr = np.asarray(b).ravel(order="F")
      # Keep first occurrence of each element in b
      u_b, first_idx = np.unique(b_arr, return_index=True)
      idx_in_ub = np.searchsorted(u_b, a_arr)
      idx_clipped = np.clip(idx_in_ub, 0, max(len(u_b) - 1, 0))
      tf = (len(u_b) > 0) & (u_b[idx_clipped] == a_arr)
      loc = np.where(tf, first_idx[idx_clipped], -1)
      return tf, loc
  ```

### `sortrows`
* **`[B, idx] = sortrows(A, cols)`**:
  ```python
  # np.lexsort sorts by the LAST key first, so reverse the column order:
  idx = np.lexsort(A[:, cols[::-1]].T)
  B = A[idx]
  ```

---

## 3. Histogramming, Binning, and `accumarray`

Horace pixel binning and projection Cuts rely heavily on `accumarray`, `histc`, and `histcounts`.

### `accumarray`
* **1D sum `accumarray(subs, val, [N, 1])`** (with 0-based `subs`):
  ```python
  # Fast 1D summation for non-negative integer bin indices:
  out = np.bincount(subs, weights=val, minlength=N)
  ```
* **N-D or general ufunc `accumarray(subs, val, sz, @fn, fillval)`**:
  ```python
  out = np.full(sz, fill_value=fillval, dtype=np.result_type(val, float))
  # For @plus / sum:
  np.add.at(out, tuple(subs.T), val)
  # For @max:
  np.maximum.at(out, tuple(subs.T), val)
  # For @min:
  np.minimum.at(out, tuple(subs.T), val)
  ```

### `histcounts` and `histc`
* **`[N, edges, bin] = histcounts(X, edges)`**:
  - `N, _ = np.histogram(X, bins=edges)`
  - For bin indices `bin`: `np.digitize(X, bins=edges, right=False)` (note: `histcounts` includes the right-most edge in the last bin `edges[-2] <= x <= edges[-1]`, whereas `np.digitize` puts `x == edges[-1]` into bin `len(edges)` unless handled).

---

## 4. Array Construction, Replication, and Broadcasting

| MATLAB Operation | NumPy Equivalent | Notes |
|---|---|---|
| `repmat(A, m, n)` | `np.tile(A, (m, n))` | Prefer implicit NumPy broadcasting over `np.tile` when used in arithmetic. |
| `repelem(v, counts)` | `np.repeat(v, counts)` | Element-wise repetition. |
| `bsxfun(@plus, A, B)` | `A + B` | NumPy broadcasts trailing dimensions automatically (ensure shapes align). |
| `cat(dim, A, B)` | `np.concatenate([A, B], axis=dim - 1)` | If `dim > A.ndim`, expand dims first or use `np.stack`. |
| `blkdiag(A, B, ...)` | `scipy.linalg.block_diag(A, B, ...)` | Block-diagonal matrix construction. |
| `circshift(A, k, dim)` | `np.roll(A, shift=k, axis=dim - 1)` | Circular shift. |
| `fliplr(A)` / `flipud(A)` | `np.fliplr(A)` / `np.flipud(A)` | Flip left-right / up-down. |

---

## 5. Linear Algebra & Trigonometry

| MATLAB Operation | NumPy / SciPy Equivalent | Notes |
|---|---|---|
| `A * B` | `A @ B` | Matrix multiplication. |
| `A .* B` | `A * B` | Element-wise multiplication. |
| `A \ B` (`mldivide`) | `np.linalg.solve(A, B)` (square) or `np.linalg.lstsq(A, B, rcond=None)[0]` | Left matrix division $A^{-1}B$. |
| `A / B` (`mrdivide`) | `np.linalg.solve(B.T, A.T).T` | Right matrix division $AB^{-1}$. |
| `sind(x)`, `cosd(x)`, `tand(x)` | `scipy.special.sindg(x)`, `cosdg(x)`, `tandg(x)` | Exact zeroes at $90^\circ, 180^\circ, 270^\circ, 360^\circ$. |
| `asind(x)`, `acosd(x)`, `atand(x)` | `np.rad2deg(np.arcsin(x))`, `np.rad2deg(np.arccos(x))`, `np.rad2deg(np.arctan(x))` | Inverse trig in degrees. |
| `atan2d(y, x)` | `np.rad2deg(np.arctan2(y, x))` | Four-quadrant inverse tangent in degrees. |
| `norm(v)` | `np.linalg.norm(v)` | Vector or matrix 2-norm. |
| `vecnorm(A, p, dim)` | `np.linalg.norm(A, ord=p, axis=dim - 1)` | Norm along specific axis. |
| `cross(a, b)` / `dot(a, b)` | `np.cross(a, b)` / `np.dot(a, b)` | 3D cross product and dot product. |

---

## 6. Cell Arrays, Structs, and Higher-Order Functions

* **Cell expansion `fn(cell_arr{:})`** -> Python star unpacking `fn(*cell_list)`
* **`cellfun(@fn, cell_arr)`** -> `[fn(x) for x in cell_list]`
* **`arrayfun(@fn, arr)`** -> `[fn(x) for x in arr]` or `np.vectorize(fn)(arr)`
* **Struct field extraction `[s_arr.field]`** -> `[s["field"] for s in s_list]` or `[getattr(obj, "field") for obj in obj_list]`
* **`isfield(S, 'name')`** -> `"name" in S` (for `dict`) or `hasattr(S, "name")` (for object)
* **`rmfield(S, 'name')`** -> `{k: v for k, v in S.items() if k != "name"}`

