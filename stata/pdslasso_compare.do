version 16.0
clear all
set more off

cd "C:\Users\rabloe\Downloads\pdslasso-20260121T160324Z-3-001\pdslasso"

* Paths (run from repo root).
local data_path "data/pdslasso_sim.csv"
local out_path "stata/pdslasso_results.csv"

* If needed: ssc install lassopack, replace

import delimited using "`data_path'", clear

* Candidate controls are x* columns.
ds x*
local controls `r(varlist)'

pdslasso y d (`controls'), loptions(c(1.1) gamma(0.05))

* Extract coefficient on treatment.
matrix b = e(b)
local d_col = colnumb(b, "d")
if (`d_col' == .) {
    di as error "Treatment variable 'd' not found in e(b) colnames."
    exit 198
}
scalar d_coef = b[1, `d_col']

* Selected controls from final regression coefficients.
local selected ""
local names : colnames b
foreach name of local names {
    if ("`name'" != "d" & "`name'" != "_cons") {
        local selected "`selected' `name'"
    }
}
local selected = trim("`selected'")
local selected_csv : subinstr local selected " " ",", all

* Write output CSV (quote selected controls to keep them in one column).
local d_coef_str = string(d_coef, "%21.12f")
file open outf using "`out_path'", write replace
file write outf "treatment_coef,selected_controls" _n
file write outf `"`d_coef_str',"`selected_csv'"' _n
file close outf
