Git Branching Model
===================

In this package, we primarily adhere to the successful Git branching model proposed by Vincent Driessen in his article, `A successful Git branching model <https://nvie.com/posts/a-successful-git-branching-model/>`_ proposed by Vincent Driessen. Here are the summary of the branches:

- **main**: The main branch hosts released software packages and is accessible only to project maintainers.

- **develop**: The develop branch serves as the main branch, where the source code at HEAD always reflects the latest development changes for the upcoming release.

- **supporting branches**: Several supporting branches exist, with three primary types for which we suggest contributors follow specific naming conventions:

    + **feature**: Feature branches are created to add new features to the software. Names should follow the format "iss[issue_number]_short_description" (e.g., "iss12_add_rotation" for issue #12, involving the addition of time series rotation). Starting with the issue number allows for easy reference to issue details when necessary. While feature branches are temporary, this naming convention helps developers navigate the codebase efficiently.

    + **hotfix**: Hotfix branches are used exclusively to fix bugs in released packages. After a bug fix, the third digit of the version number should increment by one (e.g., 1.4.2 -> 1.4.3). These branches should be prefixed with "hotfix" and followed by the upcoming version number (e.g., "hotfix_1.4.3").

    + **release**: Release branches facilitate the preparation of new production releases. They are typically not used unless the community is working on multiple releases simultaneously (e.g., pre-alpha, alpha, beta, stable, etc.).

By adhering to this Git branching model, we aim to maintain a streamlined and organized codebase, allowing for efficient collaboration and development.

