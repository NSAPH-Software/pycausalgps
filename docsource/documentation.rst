Documentation
=============
Documenting the code is an important part of contribution. There are 3 main places that needs to be addressed during contribution. 

- The main body of the documentation is written in reStrucuredText (rst) format. See this `link <https://docutils.sourceforge.io/docs/user/rst/quickref.html#hyperlink-targets>`_ for quick list of commands. 
- The codes should be provided by sufficient docstrings and each module should be added to `Modules <modules.rst>`_ file. 
- Any modification including: add, change, fix, and remove action should be added to Changelog under **Unreleased** section. 

We include private methods (starting with "_") in the documentation for easier maintenance. However, they are not meant to be used by the end-users. These parts of the documentation are generated for the developers.    

Please note that all files should be added to the **docsource** folder. 