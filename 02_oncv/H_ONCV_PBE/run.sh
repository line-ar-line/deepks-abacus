oncvpsp.x < H.dat > H.out
sed -n '/<UPF version="2.0.1">/,/<\/UPF>/p' H.out > H.upf
mv H.upf ../../H_orb/H.upf
