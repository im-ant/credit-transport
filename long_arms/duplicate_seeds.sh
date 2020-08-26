##############################################################################
# Script for duplicating a config .ini file with multiple seed
##############################################################################

#
END_IDX=3     # Up to how many file to duplciate for
MUL_BASE=3    # Base of the multiple to sed the seeds for

# Path to config file directory, CANNOT CONTAIN "/" in the end
DIR_PATH=$1

# ==
#
for cfile in $DIR_PATH/*.ini ; do
  echo $cfile

  for ((i=2;i<=END_IDX;i++)); do
    # Create the new file path
    dupfile_a="$(echo $cfile | sed 's/r1.ini/r/g')"
    dupfile_b="$dupfile_a$i"
    dupfile="$dupfile_b.ini"

    # Compute the seed
    int_seed=$((MUL_BASE*i))
    seed_line="seed = $int_seed"

    # Copy the file
    cp "$cfile" "$dupfile"

    # Change the seed line
    sed -i -e "s/seed\ =.*/$seed_line/g" "$dupfile"


    echo "    $dupfile    seed: $int_seed"
  done
done
