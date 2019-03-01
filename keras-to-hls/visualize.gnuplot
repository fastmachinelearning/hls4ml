set terminal pdfcairo enhanced dashed
set output "stress_results.pdf"

set multiplot layout 3,3
set datafile separator ','

set logscale y 2

set style line 1 \
    linecolor rgb '#0060ad' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2
set nokey

set xtics font ",4"
set ytics font ",4"

# Execution time
#set xlabel "Reuse Factor" font ",8"
set ylabel "Execute Time (secs)" font ",8" offset 2
plot "stress_results.csv" using 17:18 with linespoints linestyle 1

# DSP
#set xlabel "Reuse Factor" font ",8"
set ylabel "DSP" font ",8"
plot "stress_results.csv" using 17:11 with linespoints linestyle 1

# Best latency
#set xlabel "Reuse Factor" font ",8"
set ylabel "Best Latency" font ",8"
plot "stress_results.csv" using 17:6 with linespoints linestyle 1

# Worst latency
#set xlabel "Reuse Factor" font ",8"
set ylabel "Worst Latency" font ",8"
plot "stress_results.csv" using 17:7 with linespoints linestyle 1

# II Min
#set xlabel "Reuse Factor" font ",8"
set ylabel "II (Min)" font ",8"
plot "stress_results.csv" using 17:8 with linespoints linestyle 1

# II Max
#set xlabel "Reuse Factor" font ",8"
set ylabel "II (Max)" font ",8"
plot "stress_results.csv" using 17:9 with linespoints linestyle 1

# BRAM
#set xlabel "Reuse Factor" font ",8"
set ylabel "BRAM" font ",8"
plot "stress_results.csv" using 17:10 with linespoints linestyle 1

# FF
#set xlabel "Reuse Factor" font ",8"
set ylabel "FF" font ",8"
plot "stress_results.csv" using 17:12 with linespoints linestyle 1

# LUT
#set xlabel "Reuse Factor" font ",8"
set ylabel "LUT" font ",8"
plot "stress_results.csv" using 17:13 with linespoints linestyle 1

unset multiplot
