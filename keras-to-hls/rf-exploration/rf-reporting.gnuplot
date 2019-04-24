# Input and output file names
#CSV_FILE =  "reports/2layer_100x100.csv"
#PDF_FILE =  "reports/2layer_100x100.pdf"
#CSV_FILE = "reports/RF_stress_results_KERAS_3layer.csv"
#PDF_FILE = "reports/RF_stress_results_KERAS_3layer.pdf"
#CSV_FILE = "reports/RF_stress_results_2layer_100x100.csv"
#PDF_FILE = "reports/RF_stress_results_2layer_100x100.pdf"
#CSV_FILE =  "reports/KERAS_dense_16x200x200x200x200x200x5.csv"
#PDF_FILE =  "reports/KERAS_dense_16x200x200x200x200x200x5.pdf"
CSV_FILE = "reports/KERAS_dense_16x500x500x500x500x500x5.csv"
PDF_FILE = "reports/KERAS_dense_16x500x500x500x500x500x5.pdf"

# Set Gnuplot output on PDF file
set terminal pdfcairo enhanced dashed
set output PDF_FILE

set size 1.0, 1.0
set origin 0.0, 0.0

# CSV file (comma separated values)
set datafile separator ','
set datafile missing '-1'

# X and Y axes
set logscale y 2

set xtics font ",4"
set ytics font ",4"

set xtics border in scale 1,0.5 nomirror rotate by -35  autojustify
set ytics nomirror

set lmargin 6
set rmargin 3

#
# Line styles
#

# blue line and dots (total)
set style line 1 \
    linecolor rgb '#2166ac' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2

# green line and dots (Vivado HLS)
set style line 3 \
    linecolor rgb '#1b7637' \
    linetype 1 linewidth 1 \
    pointtype 13 pointsize 0.2

# light-green line and dots (Vivado HLS)
set style line 10 \
    linecolor rgb '#90ee90' \
    linetype 1 linewidth 1 \
    pointtype 13 pointsize 0.2

# orange line and dots (Vivado)
set style line 2 \
    linecolor rgb '#ef8a62' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2

# red thin line (timeout)
set style line 6 \
    linecolor rgb '#b2182b' \
    linetype 1 linewidth 0.1 \
    pointtype 1 pointsize 0

set style line 4 \
    linetype 0 linewidth 0.5 \
    linecolor rgb "#808080"

set style line 5 \
    linecolor rgb '#ad6000' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2

# red X
set style line 7 \
    linecolor rgb '#b2182b' \
    pointtype 2 pointsize 0.3

# green circle
set style line 8 \
    linecolor rgb '#1b7637' \
    pointtype 4 pointsize 0.2

# blue square
set style line 9 \
    linecolor rgb '#2166ac' \
    pointtype 5 pointsize 0.2

# Legend
set key font ",4"
#set key below

set grid back linestyle 4


# =============================================================================
# =============================================================================
# PAGE 1
# =============================================================================
# =============================================================================

set multiplot layout 2,2

# =============================================================================
# Success
# =============================================================================
set size 0.5,0.5
set origin 0.0,0.5 # Top-Left
set yrange [0:4]
unset logscale
set key right bottom
##set xlabel "Success" font ",8"
##set ylabel "II" font ",8"
plot CSV_FILE using ($3 == 0 && $4 == 0 ? 3 : NaN):xtic(2) title "HLS and LS Passed" with points linestyle 9, \
     CSV_FILE using ($3 == 0 ? 2 : NaN):xtic(2) title "HLS Passed" with points linestyle 8, \
     CSV_FILE using ($3 != 0 && $4 != 0 ? 1 : NaN):xtic(2) title "None" with points linestyle 7

#plot CSV_FILE using ($3 == 0 && $4 == 0 ? 3 : ($3  == 0 ? 2 : 1)):xtic(2) title "" with points linestyle 7


# =============================================================================
# HLS Time
# =============================================================================
set size 0.5,0.5
set origin 0.5,0.5 # Top-Right
set key left bottom
unset yrange
set logscale y 2
#set xlabel "Reuse Factor" font ",8"
#set ylabel "HLS Time (s)" offset 5,0 font ",8"
plot CSV_FILE using ($3 == 0 && $4 == 0 ? $5 : NaN):xtic(2) title "Total Time (s)" with linespoints linestyle 1, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $27 : NaN):xtic(2) title "HLS Time (s)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $28 : NaN):xtic(2) title "LS Time (s)" with linespoints linestyle 2, \
     CSV_FILE using 6:xtic(2) title "Timeout" with linespoints linestyle 6

# =============================================================================
# (Best and Worst) Latency
# =============================================================================
set size 0.5,0.5
set origin 0.0,0.0 # Bottom-Left
set key left top
#set xlabel "Reuse Factor" font ",8"
#set ylabel "Best Latency" font ",8"
plot CSV_FILE using ($3 == 0 && $4 == 0 ? $15 : NaN):xtic(2) title "Worst Latency" with linespoints linestyle 10, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $14 : NaN):xtic(2) title "Best Latency" with linespoints linestyle 3

# =============================================================================
# (Min and Max) II
# =============================================================================
set size 0.5,0.5
set origin 0.5,0.0 # Bottom-Right
set key left top
#set xlabel "Reuse Factor" font ",8"
#set ylabel "II" font ",8"
plot CSV_FILE using ($3 == 0 && $4 == 0 ? $16 : NaN):xtic(2) title "Min II" with linespoints linestyle 10, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $17 : NaN):xtic(2) title "Max II" with linespoints linestyle 3

unset multiplot

# =============================================================================
# =============================================================================
# PAGE 2
# =============================================================================
# =============================================================================

set multiplot layout 2,2
#title "Performance/Costs vs. Reuse Factor"

# =============================================================================
# DSP
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.0,0.5 # Top-Left
set key right
plot CSV_FILE using ($3 == 0 ? $19 : NaN):xtic(2) title "DSP (HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $24 : NaN):xtic(2) title "DSP" with linespoints linestyle 2

# =============================================================================
# FF
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.5,0.5 # Top-Right
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "FF" offset 5,0 font ",8"
plot CSV_FILE using ($3 == 0 ? $20 : NaN):xtic(2) title "FF (HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $25 : NaN):xtic(2) title "FF" with linespoints linestyle 2

# =============================================================================
# LUT
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.0,0.0 # Bottom-Left
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "LUT" offset 5,0 font ",8"
plot CSV_FILE using ($3 == 0 ? $21 : NaN):xtic(2) title "LUT (HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $26 : NaN):xtic(2) title "LUT" with linespoints linestyle 2

# =============================================================================
# BRAM
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.5,0.0 # Bottom-Right
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "BRAM" offset 5,0 font ",8"
plot CSV_FILE using ($3 == 0 ? $18 : NaN):xtic(2) title "BRAM (HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $23 : NaN):xtic(2) title "BRAM" with linespoints linestyle 2

unset multiplot
