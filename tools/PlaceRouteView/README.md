# Placement and Routing Viewer

To install the extension, copy the .vsix file to your VSCode extensions folder on your local machine: .vscode/extensions. Depending on your platform, it is located in the following folders:

Windows: %USERPROFILE%\.vscode\extensions \
macOS: ~/.vscode/extensions \
Linux: ~/.vscode/extensions 
       or
       ~/.vscode-server/extensions

Then, run the command:

```code --install-extension prviewer-1.0.0.vsix```

And the commands should be available in VSCode. You may need to enable the extension if connected to a server (connect to server -> open extensions tab -> under 'local - installed' find prviewer and install on server).

## Commands

To run the commands, press ctrl + shift + p, ensure the line starts with a ">", and type the command of your choice into the searchbar. A command will need to be run before hovering works. Files must be in .json format, as output by ```aie-translate -aie-flows-to-json``` or ```air-translate -air-herds-to-json```.

The current commands are ```Placement View: Open Placement Webview``` and ```Routing View: Open Routing Webview```.

Note that the size of the grid is determined by the largest row and column values of the switchboxes.

## Routing

The routing field must be formatted in the following way: \
``` "route<xx>": [ [ [<AIE_col #>, <AIE_Row #>], ["<direction or DMA>", ...] ], ... [] ]  ``` \
Where xx is the route number. part_test_full.json provides an example file. 

Multiple routes can be displayed at a single time in different windows. To open up a 2nd route, just hover over the name of the route you wish to display and click the link. Clicking the link again will update that route view. Saving will update all route views from a given file.

### Keywords

In order to view all of the routes given in a file, add the following route to the file:
```"route_all": []```

To view a subset of the routes, use the keyword "route_some":
```"route_some<identifier>": [<route name>, ...] ```

## Placement

The partition field must be formatted in either of the the following ways: \
``` "partition": [[<herd #>, [AIE_Row, AIE_col] ...] ...] ``` \ or \
``` "partition": [[<herd #>, "<herd>", [AIE_Row, AIE_col] ...] ...] ```  \

The herd number changes the color of the herd.
