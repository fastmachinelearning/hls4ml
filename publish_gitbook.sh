# install the plugins and build the static site
gitbook install && gitbook build

# checkout to the gh-pages branch
git checkout gh-pages

# pull the latest updates
git pull origin gh-pages --rebase

# copy the static site files into the current directory.
cp -R _book/* .

# remove 'node_modules' and '_book' directory
git clean -fx node_modules
git clean -fx _book

# add all files
git add .

# commit
git commit -a -m "update profiling example code"

# push to the origin
git push origin gh-pages

