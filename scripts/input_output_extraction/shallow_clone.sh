if [ "$#" -ne 3 ]; then
    echo "shallow-clone.sh remote commit outdir"
    exit 1
fi

remote=$1
commit=$2
outdir=$3

if [ -d $outdir ]; then
    echo "$outdir exists"
    exit 1
fi
echo "Cloning $remote commit $commit into $outdir"

mkdir -p $outdir
cd $outdir
git init
git remote add origin $remote
git fetch --depth 1 origin $commit
git checkout FETCH_HEAD

echo "Done"
