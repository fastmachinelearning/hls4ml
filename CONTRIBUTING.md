# How to Contribute

We'd love to accept your patches and contributions to this project.
There are just a few small guidelines you need to follow.

## Discussion

Share your proposal via [GitHub Issues](https://github.com/fastmachinelearning/hls4ml/issues).
If you are looking for some issues to get started with, we have a list of [good first issues](https://github.com/fastmachinelearning/hls4ml/labels/good%20first%20issue) in the issue tracker.
We also welcome submissions to improve the documentation.

## Pull Request

All submissions, including submissions by project members, require review.
We use GitHub pull requests for this purpose.
Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

1. In the pull request description, clearly document all changes made and the expected behavior.
1. If you are introducing new functionality, add at least one unit test under the `test` folder and make sure it passes before you submit the pull request.
1. Similarly, if you are fixing a bug, add at least one unit test under the `test` folder such that the master branch fails the test and your branch passes the test.
1. Install and run `pre-commit` on the files that you have edited.
1. Submit the pull request to the [main](https://github.com/fastmachinelearning/hls4ml) branch.

Pull requests are **squash-merged**, so the individual commits on your branch are not preserved.
You do not need to curate them, but the pull request title and description become the permanent record of
the change in the project history, so please write them with that in mind.

## Code Reviews

We will review your contribution and, if any additional fixes or modifications are necessary, may provide feedback to guide you.
When accepted, your pull request will be merged to the repository.

## Use of AI tools

Usage of AI tools in development for hls4ml is generally allowed. However, we require all contributors to adhere to the following guidelines:

- Contributed code must still be your own original work. It is your responsibility to make sure that the generated code is compatible with the [hls4ml license](LICENSE), [these Contributor Guidelines](CONTRIBUTING.md), and that it doesn't violate the license of either the AI tool or any third-party license obligations.
- The AI tool name and version must be disclosed in the pull request description. The pull request template has a section for this.
- Ensure you have reviewed and fully understand the generated code and be prepared to explain the reasoning behind it during review.
- AI coding agents tend to be very verbose. Please review generated code and especially comments and trim unnecessary clutter.
- Numbers, logs and test results quoted in a pull request or issue must come from runs you actually performed, not from tool output that was never executed.
- Do not spam the repository with issues and pull requests for problems that have no likelihood of appearing in real world applications. Focus instead on fixing or improving hls4ml for real users.

Do not credit an AI tool as an author of a commit. Authorship carries copyright, which a tool cannot hold, and
such trailers distort contributor statistics. Some assistants add `Co-authored-by` trailers automatically;
remove them before opening the pull request. Disclosure belongs in the pull request description, where a
reviewer will read it.

If you use an AI agent to work on hls4ml, point it at [`AGENTS.md`](AGENTS.md) in the repository root. It
states the same expectations in a form agents read, and it will save you review comments.

Submissions that appear unreviewed or copied directly from an AI tool without proper understanding may be requested to be revised or declined.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to make participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, sex characteristics, gender identity and expression,
level of experience, education, socio-economic status, nationality, personal
appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org), version 1.4,
available at https://www.contributor-covenant.org/version/1/4/code-of-conduct.html

For answers to common questions about this code of conduct, see
https://www.contributor-covenant.org/faq
