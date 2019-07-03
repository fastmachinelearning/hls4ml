# Input and output file names
#CSV_FILE =  "reports/2layer_100x100.csv"
#PDF_FILE =  "reports/2layer_100x100.pdf"
#CSV_FILE = "reports/RF_stress_results_KERAS_3layer.csv"
#PDF_FILE = "reports/RF_stress_results_KERAS_3layer.pdf"
#CSV_FILE = "reports/RF_stress_results_2layer_100x100.csv"
#PDF_FILE = "reports/RF_stress_results_2layer_100x100.pdf"
#CSV_FILE =  "reports/KERAS_dense_16x200x200x200x200x200x5.csv"
#PDF_FILE =  "reports/KERAS_dense_16x200x200x200x200x200x5.pdf"
#CSV_FILE = "reports/KERAS_dense_16x500x500x500x500x500x5.csv"
#PDF_FILE = "reports/KERAS_dense_16x500x500x500x500x500x5.pdf"
CSV_FILE = "reports/KERAS_digit_recognizer_mlp.csv"
PDF_FILE = "reports/KERAS_digit_recognizer_mlp.pdf"

# Set Gnuplot output on PDF file
set terminal pdfcairo enhanced dashed
set output PDF_FILE

set size 1.0, 1.0
set origin 0.0, 0.0

# CSV file (comma separated values)
set datafile separator ','
set datafile missing '-1'

# Common configuration for X axis
set xtics font ",4"
set xtics border in scale 1,0.5 nomirror rotate by -35  autojustify

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
set style line 11 \
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

# green empty square
set style line 8 \
    linecolor rgb '#1b7637' \
    pointtype 4 pointsize 0.2

# blue square
set style line 9 \
    linecolor rgb '#2166ac' \
    pointtype 5 pointsize 0.2

# red empty square
set style line 10 \
    linecolor rgb '#b2182b' \
    pointtype 4 pointsize 0.2


### Execution Time

# Total / blue line and dots
set style line 30 \
    linecolor rgb '#2166ac' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2

# Vivado HLS / green line and dots
set style line 31 \
    linecolor rgb '#1b7637' \
    linetype 1 linewidth 1 \
    pointtype 13 pointsize 0.2

# RTL Sim / light-green line and dots
set style line 32 \
    linecolor rgb '#90ee90' \
    linetype 1 linewidth 1 \
    pointtype 13 pointsize 0.2

# Vivado / orange line and dots
set style line 33 \
    linecolor rgb '#ef8a62' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2

# Timeout / red thin line
set style line 34 \
    linecolor rgb '#b2182b' \
    linetype 1 linewidth 0.1 \
    pointtype 1 pointsize 0

### Exit Status

# Pass (green, full square)
set style line 20 \
    linecolor rgb '#1b7637' \
    pointtype 5 pointsize 0.2

# Fail (red x)
set style line 21 \
    linecolor rgb '#b2182b' \
    pointtype 2 pointsize 0.4

# Timeout (red, full square)
set style line 22 \
    linecolor rgb '#b2182b' \
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
unset ytics
set ytics font ",6"
set ytics nomirror
set ytics("HLS" 1, "RTL Sim" 2, "Vivado" 3)
set key outside right bottom
set key horizontal
set bmargin at screen 0.64
unset logscale

plot CSV_FILE using ($3 == 2? 1 : NaN):xtic(2) notitle with points linestyle 22, \
     CSV_FILE using ($4 == 2? 2 : NaN):xtic(2) notitle with points linestyle 22, \
     CSV_FILE using ($5 == 2? 3 : NaN):xtic(2) notitle with points linestyle 22, \
     CSV_FILE using ($3 == 1? 1 : NaN):xtic(2) notitle with points linestyle 21, \
     CSV_FILE using ($4 == 1? 2 : NaN):xtic(2) notitle with points linestyle 21, \
     CSV_FILE using ($5 == 1? 3 : NaN):xtic(2) notitle with points linestyle 21, \
     CSV_FILE using ($3 == 0? 1 : NaN):xtic(2) notitle with points linestyle 20, \
     CSV_FILE using ($4 == 0? 2 : NaN):xtic(2) notitle with points linestyle 20, \
     CSV_FILE using ($5 == 0? 3 : NaN):xtic(2) notitle with points linestyle 20, \
     CSV_FILE using (NaN):xtic(2) title "Failed" with points linestyle 21, \
     CSV_FILE using (NaN):xtic(2) title "Timeout" with points linestyle 22, \
     CSV_FILE using (NaN):xtic(2) title "Passed" with points linestyle 20

unset label

# =============================================================================
# HLS Time
# =============================================================================
set size 0.5,0.5
set origin 0.5,0.5 # Top-Right
unset ytics
set ytics font ",4"
set ytics nomirror
unset yrange
set logscale y 2
set key outside right bottom
set key horizontal
set bmargin at screen 0.64

#set xlabel "Reuse Factor" font ",8"
#set ylabel "HLS Time (s)" offset 5,0 font ",8"
plot CSV_FILE using 7:xtic(2) title "Timeout" with linespoints linestyle 34, \
     CSV_FILE using ($3 == 0 && $4 == 0 && $5 == 0 ? $6 : NaN):xtic(2) title "Total Time (s)" with linespoints linestyle 30, \
     CSV_FILE using ($3 == 0 ? $32 : NaN):xtic(2) title "Vivado HLS Time (s)" with linespoints linestyle 31, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $33 : NaN):xtic(2) title "Vivado Time (s)" with linespoints linestyle 33, \
     CSV_FILE using ($3 == 0 ? $34: NaN):xtic(2) title "RTL Sim Time (s)" with linespoints linestyle 32

# =============================================================================
# (Best and Worst) Latency
# =============================================================================
set size 0.5,0.5
set origin 0.0,0.0 # Bottom-Left
set key outside right bottom
set key horizontal
set bmargin at screen 0.1

#set xlabel "Reuse Factor" font ",8"
#set ylabel "Best Latency" font ",8"
plot CSV_FILE using ($3 == 0 && $4 == 0 ? $20 : NaN):xtic(2) title "Worst Latency" with linespoints linestyle 11, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $19 : NaN):xtic(2) title "Best Latency" with linespoints linestyle 3

# =============================================================================
# (Min and Max) II
# =============================================================================
set size 0.5,0.5
set origin 0.5,0.0 # Bottom-Right
set key outside right bottom
set key horizontal
set bmargin at screen 0.1

#set xlabel "Reuse Factor" font ",8"
#set ylabel "II" font ",8"
plot CSV_FILE using ($3 == 0 && $4 == 0 ? $21 : NaN):xtic(2) title "Min II" with linespoints linestyle 11, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $22 : NaN):xtic(2) title "Max II" with linespoints linestyle 3

unset multiplot

# =============================================================================
# =============================================================================
# PAGE 2
# =============================================================================
# =============================================================================

set multiplot layout 2,2
#title "Performance/Costs vs. Reuse Factor"

## TODO Add percentages
##set y2tics font ",4"
##set y2tics nomirror
##set format y2 "%.0f%%"

# =============================================================================
# DSP
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.0,0.5 # Top-Left
set key outside right bottom
set key horizontal
set bmargin at screen 0.6

## TODO Add percentages
##stats CSV_FILE using 14 nooutput
##available_DSP = STATS_max
##stats CSV_FILE using 24
##max_DSP = STATS_max
###set y2range [0:available_DSP]
set y2range [0:100]
plot CSV_FILE using ($3 == 0 ? $24 : NaN):xtic(2) title "DSP (Vivado HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $29 : NaN):xtic(2) title "DSP (Vivado)" with linespoints linestyle 2, \
     CSV_FILE using 14:xtic(2) title "DSP (Available)" with linespoints linestyle 6

# =============================================================================
# FF
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.5,0.5 # Top-Right
set key outside right bottom
set key horizontal
set bmargin at screen 0.6

#set xlabel "Reuse Factor" font ",8"
#set ylabel "FF" offset 5,0 font ",8"
## TODO Add percentages
##stats CSV_FILE using 15 nooutput
##available_FF = STATS_max
###set y2range [0:available_FF]
##set y2range [0:100]
plot CSV_FILE using ($3 == 0 ? $25 : NaN):xtic(2) title "FF (Vivado HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $30 : NaN):xtic(2) title "FF (Vivado)" with linespoints linestyle 2, \
     CSV_FILE using 15:xtic(2) title "FF (Available)" with linespoints linestyle 6

# =============================================================================
# LUT
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.0,0.0 # Bottom-Left
set key outside right bottom
set key horizontal
set bmargin at screen 0.1

#set xlabel "Reuse Factor" font ",8"
#set ylabel "LUT" offset 5,0 font ",8"
## TODO Add percentages
##stats CSV_FILE using 16 nooutput
##available_LUT = STATS_max
###set y2range [0:available_LUT]
set y2range [0:100]
plot CSV_FILE using ($3 == 0 ? $26 : NaN):xtic(2) title "LUT (Vivado HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $31 : NaN):xtic(2) title "LUT (Vivado)" with linespoints linestyle 2, \
     CSV_FILE using 16:xtic(2) title "LUT (Available)" with linespoints linestyle 6

# =============================================================================
# BRAM
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.5,0.0 # Bottom-Right
set key outside right bottom
set key horizontal
set bmargin at screen 0.1

#set xlabel "Reuse Factor" font ",8"
#set ylabel "BRAM" offset 5,0 font ",8"
## TODO Add percentages
##stats CSV_FILE using 13 nooutput
##available_BRAM = STATS_max
###set y2range [0:available_BRAM]
set y2range [0:100]
plot CSV_FILE using ($3 == 0 ? $23 : NaN):xtic(2) title "BRAM (Vivado HLS)" with linespoints linestyle 3, \
     CSV_FILE using ($3 == 0 && $4 == 0 ? $28 : NaN):xtic(2) title "BRAM (Vivado)" with linespoints linestyle 2, \
     CSV_FILE using 13:xtic(2) title "BRAM (Available)" with linespoints linestyle 6

unset multiplot
