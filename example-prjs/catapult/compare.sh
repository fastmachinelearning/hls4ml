#! /bin/csh -f

# Compare synthesis results between Catapult and Vivado

# Inlining summary count
foreach fn (dense_latency product Op_add reduce)
  echo "INLINE $fn ============================================="
  echo "  Vivado count"
  grep 'Inlining ' my-Vivado-test/vivado_hls.log | sed -e 's/ into .*//' | grep 'nnet::' | grep $fn | wc -l
  echo "  Catapult count"
  grep " Inlining " my-Catapult-test/catapult.log | sed -e 's/.*: //' | grep 'nnet::' | grep $fn | wc -l
end
# Inlining details
foreach fn (dense_latency product Op_add reduce)
  echo "  Vivado details"
  grep 'Inlining ' my-Vivado-test/vivado_hls.log | sed -e 's/ into .*//' | grep 'nnet::' | grep $fn
  echo "  Catapult details"
  grep " Inlining " my-Catapult-test/catapult.log | sed -e 's/.*: //' | grep 'nnet::' | grep $fn
end

# Loop unrolling
echo "UNROLLING ============================================="
grep "Unrolling loop " my-Vivado-test/vivado_hls.log | sed -e 's/INFO.* Unrolling loop \(.*\) .* in function \(.*\) .* factor of \(.*\)/Unrolling loop \1 \2 factor \3/p' | sort -u | wc -l
grep -i 'unrolled' my-Catapult-test/catapult.log | sed -e 's/.* Loop \(.*\) is being \(.*\) unrolled \(.*\)/Loop \1 \2 \3/' | sort -u | wc -l
echo "  Vivado details"
grep "Unrolling loop " my-Vivado-test/vivado_hls.log | sed -e 's/INFO.* Unrolling loop \(.*\) .* in function \(.*\) .* factor of \(.*\)/Unrolling loop \1 \2 factor \3/p' | sort -u
echo "  Catapult details"
grep -i 'unrolled' my-Catapult-test/catapult.log | sed -e 's/.* Loop \(.*\) is being \(.*\) unrolled \(.*\)/Loop \1 \2 \3/' | sort -u

