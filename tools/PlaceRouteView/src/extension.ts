// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

'use strict';

import * as vscode from 'vscode';
import {json} from 'stream/consumers';
import {stringify} from 'querystring';
import {open} from 'fs';

const CSS_COLOR_NAMES = [
  'purple',
  'cyan',
  'yellow',
  'orange',
  'blue',
  'pink',
  'red',
  "Aqua",
  "BlueViolet",
  "Brown",
  "BurlyWood",
  "CadetBlue",
  "Chartreuse",
  "Chocolate",
  "Coral",
  "CornflowerBlue",
  "Crimson",
  "DarkBlue",
  "DarkCyan",
  "BlanchedAlmond",
  "Aquamarine",
  "DarkGoldenRod",
  "DarkGray",
  "DarkGrey",
  "DarkGreen",
  "DarkKhaki",
  "DarkMagenta",
  "DarkOliveGreen",
  "DarkOrange",
  "DarkOrchid",
  "DarkRed",
  "DarkSalmon",
  "DarkSeaGreen",
  "DarkSlateBlue",
  "DarkSlateGray",
  "DarkSlateGrey",
  "DarkTurquoise",
  "DarkViolet",
  "DeepPink",
  "DeepSkyBlue",
  "DimGray",
  "DimGrey",
  "DodgerBlue",
  "FireBrick",
  "FloralWhite",
  "ForestGreen",
  "Fuchsia",
  "Gold",
  "GoldenRod",
  "Gray",
  "Grey",
  "GreenYellow",
  "HoneyDew",
  "HotPink",
  "IndianRed",
  "Indigo",
  "Khaki",
  "Lavender",
  "LavenderBlush",
  "LawnGreen",
  "LightBlue",
  "LightCoral",
  "LightCyan",
  "LightGray",
  "LightGrey",
  "LightGreen",
  "LightPink",
  "LightSalmon",
  "LightSeaGreen",
  "LightSkyBlue",
  "LightSlateGray",
  "LightSlateGrey",
  "LightSteelBlue",
  "Lime",
  "LimeGreen",
  "Magenta",
  "Maroon",
  "MediumAquaMarine",
  "MediumBlue",
  "MediumOrchid",
  "MediumPurple",
  "MediumSeaGreen",
  "MediumSlateBlue",
  "MediumSpringGreen",
  "MediumTurquoise",
  "MediumVioletRed",
  "MidnightBlue",
  "MintCream",
  "MistyRose",
  "Moccasin",
  "NavajoWhite",
  "Navy",
  "Olive",
  "OliveDrab",
  "OrangeRed",
  "Orchid",
  "PaleGoldenRod",
  "PaleGreen",
  "PaleTurquoise",
  "PaleVioletRed",
  "PeachPuff",
  "Peru",
  "Plum",
  "PowderBlue",
  "RebeccaPurple",
  "Red",
  "RosyBrown",
  "RoyalBlue",
  "SaddleBrown",
  "Salmon",
  "SandyBrown",
  "SeaGreen",
  "Sienna",
  "Silver",
  "SkyBlue",
  "SlateBlue",
  "SlateGray",
  "SlateGrey",
  "SpringGreen",
  "SteelBlue",
  "Tan",
  "Teal",
  "Thistle",
  "Tomato",
  "Turquoise",
  "Violet",
  "Wheat",
  "YellowGreen",
];

var row_size = -1;
var col_size = -1;

export function activate(context: vscode.ExtensionContext) {
  // Partition pannel
  let openFilePartPanel: any = {};
  let currPartFileName: string;

  // Routing pannel
  let openFileRoutePanel: any = {};
  let currRouteFileName: string;

  context.subscriptions.push(
      // sets up the webview (visual representation of a partition)
      vscode.commands.registerCommand('prviewer.start', (passedInPart: string|
                                                         undefined) => {
        const editor = vscode.window.activeTextEditor;
        if (editor === undefined) {
          vscode.window.showInformationMessage(
              `Try running the command again with a .json file \
				as the active window.`);
          return;
        }
        if (!editor.document.fileName.endsWith(".json")) {
          vscode.window.showInformationMessage(`
				Try running the command again with a .json file as the
				active window.`);
          return;
        }
        const document = editor.document;
        currPartFileName = document.fileName;
        if (passedInPart && editor.document.fileName in openFilePartPanel &&
            passedInPart in openFilePartPanel[currPartFileName]) {
          openFilePartPanel[currPartFileName][passedInPart].reveal(
              vscode.ViewColumn.Beside);
        }
        // get word within selection
        const json_file = document.getText();
        var json_object: object = {};

        // Ensure valid JSON
        try {
          json_object = JSON.parse(json_file);
        } catch (SyntaxError) {
          vscode.window.showInformationMessage("Invalid .json syntax.");
          return;
        }

        // if we're entering from a new hover event
        if (passedInPart &&
            !(openFilePartPanel.hasOwnProperty(currPartFileName))) {
          const panel = vscode.window.createWebviewPanel(
              'prviewer',
              `${passedInPart} in ` +
                  currPartFileName.substring(currPartFileName.lastIndexOf("/") +
                                             1),
              vscode.ViewColumn.Beside, {
                // Enable scripts in the webview
                enableScripts : true
              });

          if (!openFilePartPanel.hasOwnProperty(currPartFileName)) {
            openFilePartPanel[currPartFileName] = {};
          }
          openFilePartPanel[currPartFileName][passedInPart] = panel;

          panel.onDidDispose(() => {
            // When the panel is closed, cancel any future updates to the
            // webview content
            delete openFilePartPanel[currPartFileName][passedInPart];
          }, null, context.subscriptions);

          update_window(document,
                        openFilePartPanel[currPartFileName][passedInPart],
                        false, true, undefined, json_object, passedInPart);
          return;
        }

        // if we're entering from an already opened hover event or a save event
        if (passedInPart &&
            (openFilePartPanel.hasOwnProperty(currPartFileName)) &&
            openFilePartPanel[currPartFileName].hasOwnProperty(passedInPart)) {
          update_window(document,
                        openFilePartPanel[currPartFileName][passedInPart],
                        false, true, undefined, json_object, passedInPart);
          return;
        }

        // if we're entering from a hover event and the route is not currently
        // being displayed
        if (passedInPart &&
            (openFilePartPanel.hasOwnProperty(currPartFileName)) &&
            !openFilePartPanel[currPartFileName].hasOwnProperty(passedInPart)) {
          const panel = vscode.window.createWebviewPanel(
              'prviewer',
              `${passedInPart} in ` +
                  currPartFileName.substring(currPartFileName.lastIndexOf("/") +
                                             1),
              vscode.ViewColumn.Beside, {
                // Enable scripts in the webview
                enableScripts : true
              });

          if (!openFilePartPanel.hasOwnProperty(currPartFileName)) {
            openFilePartPanel[currPartFileName] = {};
          }
          openFilePartPanel[currPartFileName][passedInPart] = panel;

          panel.onDidDispose(() => {
            // When the panel is closed, cancel any future updates to the
            // webview content
            delete openFilePartPanel[currPartFileName][passedInPart];
          }, null, context.subscriptions);
          update_window(document,
                        openFilePartPanel[currPartFileName][passedInPart],
                        false, true, undefined, json_object, passedInPart);
          return;
        }

        // if we're entering from a ctrl + shift + p event (first time)
        // haven't passed a Part in, and the file isn't open
        if (!passedInPart) {

          const drawName = getFirstPart(json_object);
          if (!drawName) {
            vscode.window.showInformationMessage(
                "Please add a partition to the file before running the Part command.");
            return;
          }

          if (openFilePartPanel.hasOwnProperty(currPartFileName) &&
              openFilePartPanel[currPartFileName].hasOwnProperty(drawName)) {
            update_window(document,
                          openFilePartPanel[currPartFileName][drawName], false,
                          true, undefined, json_object, drawName);
            return;
          }
          const panel = vscode.window.createWebviewPanel(
              'prviewer',
              `${drawName} in ` + currPartFileName.substring(
                                      currPartFileName.lastIndexOf("/") + 1),
              vscode.ViewColumn.Beside, {
                // Enable scripts in the webview
                enableScripts : true
              });

          if (!openFilePartPanel.hasOwnProperty(currPartFileName)) {
            openFilePartPanel[currPartFileName] = {};
          }
          openFilePartPanel[currPartFileName][drawName] = panel;

          panel.onDidDispose(() => {
            // When the panel is closed, cancel any future updates to the
            // webview content
            delete openFilePartPanel[currPartFileName][drawName];
          }, null, context.subscriptions);

          update_window(document, openFilePartPanel[currPartFileName][drawName],
                        false, true, undefined, json_object, drawName);
          return;
        }
      }));

  // Option to hover over the word, and when clicked, show the desired net
  vscode.languages.registerHoverProvider(
      'json', new (class implements vscode.HoverProvider {
        provideHover(_document: vscode.TextDocument, _position: vscode.Position,
                     _token: vscode.CancellationToken):
            vscode.ProviderResult<vscode.Hover> {
          const range = _document.getWordRangeAtPosition(_position);
          const word = _document.getText(range);
          if (word.includes("route")) {
            const commentCommandUri =
                vscode.Uri.parse(`command:prviewer.drawRoutes?${
                    encodeURIComponent(JSON.stringify(word))}`);
            const contents = new vscode.MarkdownString(
                `[Display ${word}](${commentCommandUri})`);

            contents.isTrusted = true;
            return new vscode.Hover(contents);
          }
        }
      })());

  // Show the desired partition when hovering
  vscode.languages.registerHoverProvider(
      'json', new (class implements vscode.HoverProvider {
        provideHover(_document: vscode.TextDocument, _position: vscode.Position,
                     _token: vscode.CancellationToken):
            vscode.ProviderResult<vscode.Hover> {
          const range = _document.getWordRangeAtPosition(_position);
          const word = _document.getText(range);
          if (word.includes("part")) {
            const commentCommandUri =
                vscode.Uri.parse(`command:prviewer.start?${
                    encodeURIComponent(JSON.stringify(word))}`);
            const contents = new vscode.MarkdownString(
                `[Display ${word}](${commentCommandUri})`);

            contents.isTrusted = true;
            return new vscode.Hover(contents);
          }
        }
      })());

  context.subscriptions.push(
      // sets up the webview (visual representation of a partition)
      vscode.commands.registerCommand('prviewer.drawRoutes', (passedInRoute:
                                                                  string|
                                                              undefined) => {
        const editor = vscode.window.activeTextEditor;
        if (editor === undefined) {
          vscode.window.showInformationMessage(`
				Try running the command again with a .json file as the
				active window.`);
          return;
        }
        if (!editor.document.fileName.endsWith(".json")) {
          vscode.window.showInformationMessage(`
				Try running the command again with a .json file as the
				active window.`);
          return;
        }
        const document = editor.document;
        currRouteFileName = document.fileName;
        if (passedInRoute && editor.document.fileName in openFileRoutePanel &&
            passedInRoute in openFileRoutePanel[currRouteFileName]) {
          openFileRoutePanel[currRouteFileName][passedInRoute].reveal(
              vscode.ViewColumn.Beside);
        }
        // get word within selection
        const json_file = document.getText();
        var json_object: object = {};

        // Ensure valid JSON
        try {
          json_object = JSON.parse(json_file);
        } catch (SyntaxError) {
          vscode.window.showInformationMessage("Invalid .json syntax.");
          return;
        }

        // if we're entering from a new hover event
        if (passedInRoute &&
            !(openFileRoutePanel.hasOwnProperty(currRouteFileName))) {
          const panel = vscode.window.createWebviewPanel(
              'prviewer',
              `${passedInRoute} in ` +
                  currRouteFileName.substring(
                      currRouteFileName.lastIndexOf("/") + 1),
              vscode.ViewColumn.Beside, {
                // Enable scripts in the webview
                enableScripts : true
              });

          if (!openFileRoutePanel.hasOwnProperty(currRouteFileName)) {
            openFileRoutePanel[currRouteFileName] = {};
          }
          openFileRoutePanel[currRouteFileName][passedInRoute] = panel;

          panel.onDidDispose(() => {
            // When the panel is closed, cancel any future updates to the
            // webview content
            delete openFileRoutePanel[currRouteFileName][passedInRoute];
          }, null, context.subscriptions);

          update_window(document,
                        openFileRoutePanel[currRouteFileName][passedInRoute],
                        true, false, passedInRoute, json_object, undefined);
          return;
        }

        // if we're entering from an already opened hover event or a save event
        if (passedInRoute &&
            (openFileRoutePanel.hasOwnProperty(currRouteFileName)) &&
            openFileRoutePanel[currRouteFileName].hasOwnProperty(
                passedInRoute)) {
          update_window(document,
                        openFileRoutePanel[currRouteFileName][passedInRoute],
                        true, false, passedInRoute, json_object, undefined);
          return;
        }

        // if we're entering from a hover event and the route is not currently
        // being displayed
        if (passedInRoute &&
            (openFileRoutePanel.hasOwnProperty(currRouteFileName)) &&
            !openFileRoutePanel[currRouteFileName].hasOwnProperty(
                passedInRoute)) {
          const panel = vscode.window.createWebviewPanel(
              'prviewer',
              `${passedInRoute} in ` +
                  currRouteFileName.substring(
                      currRouteFileName.lastIndexOf("/") + 1),
              vscode.ViewColumn.Beside, {enableScripts : true});

          if (!openFileRoutePanel.hasOwnProperty(currRouteFileName)) {
            openFileRoutePanel[currRouteFileName] = {};
          }
          openFileRoutePanel[currRouteFileName][passedInRoute] = panel;

          panel.onDidDispose(() => {
            // When the panel is closed, cancel any future updates to the
            // webview content
            delete openFileRoutePanel[currRouteFileName][passedInRoute];
          }, null, context.subscriptions);
          update_window(document,
                        openFileRoutePanel[currRouteFileName][passedInRoute],
                        true, false, passedInRoute, json_object, undefined);
          return;
        }

        // if we're entering from a ctrl + shift + p event (first time)
        // haven't passed a route in, and the file isn't open
        if (!passedInRoute) {

          const drawName = getFirstRoute(json_object);
          if (!drawName) {
            vscode.window.showInformationMessage(
                "Please add a route to the file before running the route command.");
            return;
          }

          if (openFileRoutePanel.hasOwnProperty(currRouteFileName) &&
              openFileRoutePanel[currRouteFileName].hasOwnProperty(drawName)) {
            update_window(document,
                          openFileRoutePanel[currRouteFileName][drawName], true,
                          false, drawName, json_object, undefined);
            return;
          }
          const panel = vscode.window.createWebviewPanel(
              'prviewer',
              `${drawName} in ` + currRouteFileName.substring(
                                      currRouteFileName.lastIndexOf("/") + 1),
              vscode.ViewColumn.Beside, {enableScripts : true});

          if (!openFileRoutePanel.hasOwnProperty(currRouteFileName)) {
            openFileRoutePanel[currRouteFileName] = {};
          }
          openFileRoutePanel[currRouteFileName][drawName] = panel;

          panel.onDidDispose(() => {
            // When the panel is closed, cancel any future updates to the
            // webview content
            delete openFileRoutePanel[currRouteFileName][drawName];
          }, null, context.subscriptions);

          update_window(document,
                        openFileRoutePanel[currRouteFileName][drawName], true,
                        false, drawName, json_object, undefined);
          return;
        }
      }));

  // Updates the webview if we have an open webview, and it was saved.
  context.subscriptions.push(vscode.workspace.onDidSaveTextDocument(e => {
    if (!openFilePartPanel && !openFileRoutePanel) {
      return;
    }
    if (vscode.window.activeTextEditor?.document.fileName == undefined ||
        (!(vscode.window.activeTextEditor.document.fileName in
           openFilePartPanel) &&
         !(vscode.window.activeTextEditor.document.fileName in
           openFileRoutePanel))) {
      return;
    }

    // get word within selection
    const json_file = e.getText();
    var json_object: object = {};

    try {
      json_object = JSON.parse(json_file);
    } catch (SyntaxError) {
      vscode.window.showInformationMessage("Invalid .json syntax.");
      return;
    }

    if (e.fileName in openFileRoutePanel) {
      const keys = Object.keys(openFileRoutePanel[e.fileName])
      for (var i = 0; i < keys.length; i++) {
        const route = keys[i];
        vscode.commands.executeCommand('prviewer.drawRoutes', route);
      }
    }

    if (e.fileName in openFilePartPanel) {
      const keys = Object.keys(openFilePartPanel[e.fileName])
      for (var i = 0; i < keys.length; i++) {
        const part = keys[i];
        vscode.commands.executeCommand('prviewer.start', part);
      }
    }
  }));
}

// Draws the webview window
function update_window(document: vscode.TextDocument,
                       panel: vscode.WebviewPanel, isRoute: boolean,
                       isPart: boolean, currRouteName: string|undefined,
                       json_object: any, currPartName: string|undefined) {

  const switchbox_dict_keys: Array<Array<string>> = sort_keys(json_object);
  // if we don't have a partition field
  if (isPart && switchbox_dict_keys[1].length === 0) {
    vscode.window.showInformationMessage(`No partitioning field was found.
		Please ensure that the .json file has a valid partitioning field in the
		form: "partition<xx>": [[<partition#>, <partition name>, [<row#>, <col#>], ...], ...]`);
    return;
  }
  if (isRoute && switchbox_dict_keys[2].length === 0) {
    vscode.window.showInformationMessage(`No routing field was found.
		Please ensure that the .json file has a valid partitioning field in the
		form: "Route<xx>": [[[AIE_col, AIE_row], [<Direction or DMA>, ...]] ... [] ]`);
  }

  // Get the data into dictionaries / arrays of their own type
  const switchbox_dict: {[key: string]: object} =
      getSwitchboxes(json_object, switchbox_dict_keys[0]);
  row_size = get_max_key_value(switchbox_dict, "row") + 1;
  col_size = get_max_key_value(switchbox_dict, "col") + 1;

  if (row_size < 1 || col_size < 1) {
    vscode.window.showInformationMessage(
        "Row and / or column size defined to be less than 1.\
		   Please ensure that the switchbox fields have the appropriate row and column entries.");
    return;
  }

  if (isPart) {
    var partition_array;
    if (typeof (currPartName) === 'undefined') {
      partition_array = get_partition(json_object, switchbox_dict_keys[1]);
      panel.title =
          `${switchbox_dict_keys[1][0]} in ` +
          document.fileName.substring(document.fileName.lastIndexOf("/") + 1);
    } else {
      partition_array = get_partition(json_object, [ currPartName ]);
    }
    const format_check_part =
        check_partition_array_validity(partition_array, row_size, col_size);
    if (format_check_part) {
      vscode.window.showInformationMessage(format_check_part);
      return;
    }
    const sorted_partiton_array = rearrange_partition(partition_array);
    const colored_grid = color_grid(sorted_partiton_array, row_size, col_size);
    var region_array: Array<number> = [];

    if (switchbox_dict_keys[3].length > 0) {
      region_array = get_region(json_object, switchbox_dict_keys[3]);
      const format_check_region = check_region_array_validity(region_array);
      if (format_check_region) {
        vscode.window.showInformationMessage(format_check_region);
        return;
      }
    }
    panel.webview.html =
        getWebviewContent(colored_grid, region_array, undefined, -1);
    return;
  }
  var number_of_routes = -1;
  if (isRoute) {
    var routes: any = [];
    if (typeof (currRouteName) === 'undefined') {
      number_of_routes = 1;
      routes = getRoutes(json_object, switchbox_dict_keys[2]);
      panel.title =
          `${switchbox_dict_keys[2][0]} in ` +
          document.fileName.substring(document.fileName.lastIndexOf("/") + 1);
    } else if (currRouteName === 'route_all') {
      var new_keys: Array<string> = [];
      for (var i = 0; i < switchbox_dict_keys[2].length; i++) {
        if (switchbox_dict_keys[2][i].includes("all") ||
            switchbox_dict_keys[2][i].includes("some")) {
          continue;
        } else {
          new_keys.push(switchbox_dict_keys[2][i]);
        }
      }
      routes = getRoutes(json_object, new_keys);
      number_of_routes = new_keys.length;
    } else if (currRouteName.includes('route_some')) {
      var routes_to_route = getRoutes(json_object, [ currRouteName ]);
      routes_to_route = routes_to_route.flat();
      var index = -1;
      var test = 0;
      // Recursive look for 'route_some'
      while ((index = match(routes_to_route, 'route_some')) >= 0) {
        const new_routes =
            getRoutes(json_object, [ routes_to_route[index] ]).flat();
        routes_to_route.splice(index, 1);
        routes_to_route.push(new_routes);
        routes_to_route = routes_to_route.flat();
        test += 1;
        if (test > 10000) {
          vscode.window.showInformationMessage(
              "Something went wrong when trying to place nested routes. Please be a bit more gentle :)");
          return;
        }
      }
      routes_to_route = [...new Set(routes_to_route) ];
      routes = getRoutes(json_object, routes_to_route);
      number_of_routes = routes_to_route.length;
    } else {
      number_of_routes = 1;
      routes = getRoutes(json_object, [ currRouteName ]);
    }
    for (var i = 0; i < routes.length; i++) {
      const error_check =
          check_route_array_validity([ routes[i] ], row_size, col_size);
      if (error_check) {
        vscode.window.showInformationMessage(error_check);
        return;
      }
    }
    panel.webview.html =
        getWebviewContent(undefined, [], routes, number_of_routes);
  }
}

// Gets first instance of route name in .json object
function getFirstRoute(json_object: any) {
  for (var key in json_object) {
    if (key.includes("route")) {
      return key;
    }
  }
  return;
}

function match(array: any, str: string) {
  for (var i = 0; i < array.length; i++) {
    if (array[i].includes(str)) {
      return i;
    }
  }
  return -1;
}

// Gets first instance of route name in .json object
function getFirstPart(json_object: any) {
  for (var key in json_object) {
    if (key.includes("partition")) {
      return key;
    }
  }
  return;
}

// Gets the desired routes from a .json file
// Example:
//   json_file = ["route01": <some_attributes>, "route02": <some_attributes>,
//               "router01": <some_attributes>]
//   keys = ["route01", "route02"] ->
//   target_elements = ["route01": <some_attributes>, "route02"
//   <some_attributes>]
// Inputs:
//   json_file (Object): parsed JSON file
//   keys (Array<string>): keys to be added to the final dictionary
// Returns:
//   target_elements (Array): Array containing all of the routes
function getRoutes(json_file: any, keys: Array<string>) {
  var target_elements = [];
  for (var i = 0; i < keys.length; i++) {
    target_elements[i] = json_file[keys[i]];
  }
  return target_elements;
}

// Gets the desired element from the parsed AIE JSON file.
// Example:
//   json_file = ["switchbox01": <some_attributes>, "switchbox02":
//   <some_attributes>,
//               "router01": <some_attributes>]
//   keys = ["switchbox01", "switchbox02"] ->
//   target_elements = ["switchbox01": <some_attributes>, "switchbox02"
//   <some_attributes>]
// Inputs:
//   json_file (Object): parsed JSON file
//   keys (Array<string>): keys to be added to the final dictionary
// Returns:
//   target_elements {[key: string]: object}: dict containing objects of AIE
//   elements
function getSwitchboxes(json_file: any, keys: Array<string>) {
  var target_elements: {[key: string]: object} = {};
  for (var i = 0; i < keys.length; i++) {
    target_elements[keys[i]] = json_file[keys[i]];
  }
  return target_elements;
}

// get_partition
function get_partition(json_file: any, keys: Array<string>) {
  const target_elements = json_file[keys[0]];
  return target_elements;
}

// get region
function get_region(json_file: any, keys: Array<string>) {
  const target_elements = json_file[keys[0]];
  return target_elements;
}

// Gets the max value of a field from a dictionary
function get_max_key_value(element_list: any, value: string) {
  var max_val = -1;
  Object.keys(element_list).forEach(function(key) {
    if (element_list[key][value] > max_val) {
      max_val = element_list[key][value]
    }
  });
  return max_val;
}

// Finds the relevant keys in a file, and returns them as an array
// Example:
//   json_list = ["switchboxes01", "routes01", "switchboxes02"] ->
//   [["switchboxes01", "switchboxes02"], ["routes01"]]
// Inputs:
//   json_list (object): Parsed JSON object
// Returns:
//   all_sorted_labels (array): 2d Array containing arrays of relevant key names
//                              e.g. [[<switchbox names>], [<partitions>],
//                              [<routes>], [<regions>]]
function sort_keys(json_list: Object) {
  const obj_keys = Object.keys(json_list);
  var switchboxes = [];
  var route_routes = [];
  var partitions = [];
  var regions = [];
  for (var i = 0; i < obj_keys.length; i++) {
    if (obj_keys[i].includes("switchbox")) {
      switchboxes.push(String(obj_keys[i]));
    } else if (obj_keys[i].includes("route")) {
      route_routes.push(obj_keys[i]);
    } else if (obj_keys[i].includes("partition")) {
      partitions.push(obj_keys[i]);
    } else if (obj_keys[i].includes("region")) {
      regions.push(obj_keys[i]);
    }
  }
  var all_sorted_labels = [];
  all_sorted_labels.push(switchboxes);
  all_sorted_labels.push(partitions);
  all_sorted_labels.push(route_routes);
  all_sorted_labels.push(regions);
  return all_sorted_labels;
}

// Ensures that the array of values we have is valid for routes
//
// Examples:
//   with max_row = 3; max_col = 3
//   [ [[0, 1], ["North"]], [] ] - valid
//   [ [[0, 1], ["North"]] ] - invalid
//   [ [[0, 1], ["North", "North"]], [] ] - invalid
//   [ [[0, 1], ["East", "North"]], [[0, 2], ["East"]], [[1, 1] ["DMA"]], []] -
//   valid [ [[0, 1], ["East", "North"]], [[0, 2], ["North"]], [[1, 1] ["DMA"]],
//   []] - invalid
// Inputs:
//   max_row (integer): the number of rows
//   max_col (integer): the number of columns
//   partition_array (object): Array of the partitions in the form:
//                             [ [[<AIE_col #>, <AIE_row #>], ["<direction or
//                             DMA>", ...]], ... [] ]
// Outputs:
//   null if valid, string if invalid
function check_route_array_validity(route_array: any, max_row: number,
                                    max_col: number) {
  // valid array dimensions
  var used_tile_locations = [];
  // assert(route_array.length > 0);
  if (route_array[0].length === 0) {
    return "No data in route array 0"
  }
  for (var i = 0; i < route_array[0].length; i++) {
    // last element in the json file is [].
    if (i === route_array[0].length - 1) {
      if (route_array[0][i].length === 0) {
        break;
      } else {
        return "Ensure that the last element in a route is []."
      }
    }
    // col and row coords
    if (route_array[0][i].length !== 2) {
      return "More than 2 elements in a route segment found: " +
             String(route_array[0][i]) + "\
			. Please use format [ [[AIE_col, AIE_row], ['<direction or DMA or Core>', ...]] ... [] ]"
    }
    // make sure we have locations or dma for the row and col coords
    if (route_array[0][i][0].length !== 2) {
      return "Incorrect number of entries in tile " + String(i);
    }
    if (typeof (route_array[0][i][0][1]) !== 'number' ||
        typeof (route_array[0][i][0][0]) !== 'number') {
      return "Entry in tile [" + String(route_array[0][i][0][1]) + ", " +
             String(route_array[0][i][0][1]) + "]\
				        is not a number";
    }
    // ensure we're within the bounds of the switchboxes that were provided in
    // the .json
    if (route_array[0][i][0][0] >= max_col ||
        route_array[0][i][0][1] >= max_row || route_array[0][i][0][0] < 0 ||
        route_array[0][i][0][1] < 0) {
      return "Tile coordinates are outside of max range in tile \
				[" +
             String(route_array[0][i][0][0]) + ", " +
             String(route_array[0][i][0][1] + "].");
    }
    // Ensure no repeat tiles
    var a = JSON.stringify(used_tile_locations);
    var b = JSON.stringify(route_array[0][i][0]);
    if (a.indexOf(b) !== -1) {
      return "Repeat use of tile assignment: " + String(b);
    }
    used_tile_locations.push(route_array[0][i][0]);
    // only one of each coordinate used in a single route
    var north = 0;
    var south = 0;
    var east = 0;
    var west = 0;
    var dma = 0;
    for (var j = 0; j < route_array[0][i][1].length; j++) {
      if (route_array[0][i][1][j] !== "North" &&
          route_array[0][i][1][j] !== "East" &&
          route_array[0][i][1][j] !== "South" &&
          route_array[0][i][1][j] !== "West" &&
          route_array[0][i][1][j] !== "DMA" &&
          route_array[0][i][1][j] !== "Core") {
        return "Value of " + route_array[0][i][1][j] + " in route 0, tile [" +
               "\
				" +
               route_array[0][i][0][0] + ", " + route_array[0][i][0][1] +
               "] is not valid. \
					Please use one of 'North', 'East', 'South', 'West', 'Core', or 'DMA'. ";
      }
      if (route_array[0][i][1][j] === "North") {
        if (route_array[0][i][0][1] === max_row - 1) {
          return "Attempting to draw out of bounds in tile [" +
                 String(route_array[0][i][0][0]) + "\
					, " +
                 String(route_array[0][i][0][1]) + "]."
        }
        north += 1;
      }
      if (route_array[0][i][1][j] === "East") {
        if (route_array[0][i][0][0] === max_col - 1) {
          return "Attempting to draw out of bounds in tile [" +
                 String(route_array[0][i][0][0]) + "\
					, " +
                 String(route_array[0][i][0][1]) + "]."
        }
        east += 1;
      }
      // Can go south of row 0 because of the shim
      if (route_array[0][i][1][j] === "South") {
        south += 1;
      }
      if (route_array[0][i][1][j] === "West") {
        if (route_array[0][i][0][0] === 0) {
          return "Attempting to draw out of bounds in tile [" +
                 String(route_array[0][i][0][0]) + "\
					, " +
                 String(route_array[0][i][0][1]) + "]."
        }
        west += 1;
      }
      if (route_array[0][i][1][j] === "DMA") {
        dma += 1;
      }
      if (north > 1 || east > 1 || west > 1 || south > 1 || dma > 1) {
        return "Duplicate coordinate directions or DMA access found at \
				tile [" +
               String(route_array[0][i][0][0]) + ", " +
               String(route_array[0][i][0][1]) + "] \
				 (" +
               route_array[0][i][1][j] + ").";
      }
    }
  }
  return null;
}

// Ensures that the array of values we have is valid for partitions
//
// Examples:
//   [ [2, [1, 1], [0, 1]], [1, [1, 0]] ] - valid
//   [ [2, [1, 1,]], [2, [1, 0]] ] - invalid
//   [ [1, [2, 1]], [2, [2, 1], [1, 1]] ] - invalid
// Inputs:
//   max_row (integer): the number of rows
//   max_col (integer): the number of columns
//   partition_array (object): Array of the partitions in the form [ [Partititon
//   #, [AIE_row, AIE_col]] ]
// Outputs:
//   null if valid, string if invalid
function check_partition_array_validity(partition_array: any, max_row: number,
                                        max_col: number) {
  // valid array dimensions
  // var used_partitions: Array<number> = [];
  var used_tile_locations = [];

  for (var i = 0; i < partition_array.length; i++) {
    if (typeof (partition_array[i][0]) !== 'number') {
      return "invalid partition number: " + String(partition_array[i][0]);
    }
    if (partition_array[i].length > 1) {
      var j = 1;
      if (typeof (partition_array[i][1]) == 'string') {
        var j = 2;
      } else if (typeof (partition_array[i][1]) !== 'object') {
        return "Incorrect string value: the second value in a partition must be either a string\
				        or a list."
      }
      for (j; j < partition_array[i].length; j++) {
        var a = JSON.stringify(used_tile_locations);
        var b = JSON.stringify(partition_array[i][j]);
        if (a.indexOf(b) !== -1) {
          return "Repeat use of tile assignment: " + String(b);
        }
        used_tile_locations.push(partition_array[i][j]);
        if (typeof (partition_array[i][j]) !== 'object' ||
            partition_array[i][j].length !== 2) {
          return "Invalid [x, y] array at partition " + String(i) + ", entry " +
                 String(j);
        }
        if (typeof (partition_array[i][j][0]) !== 'number' ||
            typeof (partition_array[i][j][1]) !== 'number') {
          return "Arguments in [x, y] tile are not integers at partition " +
                 String(i) + ", entry " + String(j);
        }
        if (partition_array[i][j][0] >= max_row ||
            partition_array[i][j][1] >= max_col ||
            partition_array[i][j][0] < 0 || partition_array[i][j][1] < 0) {
          return "Tile coordinates are outside of max range in partition " +
                 String(i) + ", entry " + String(j);
        }
      }
    }
  }
  return null;
}

// Ensures array of region values is acceptable
function check_region_array_validity(region_array: Array<number>) {
  if (region_array.length !== 2) {
    return "Invalid length of region array. Please use the format [<color number>, <# AIEs per row \
			of partition>, <# AIEs per column of partition>]."
  }
  if (typeof (region_array[0]) !== 'number' ||
      typeof (region_array[1]) !== 'number') {
    return "Invalid length of region array. Please use the format [<color number>, <# AIEs per row \
			of partition>, <# AIEs per column of partition>]."
  }
  if (region_array.length === 2 && row_size % region_array[0] !== 0 ||
      col_size % region_array[1] !== 0) {
    return "Please ensure region works given the grid dimensions."
  }
}

// Translates the partitoned AIEs to an array of colors
// Example:
//   partitions: [[1, 0, 1], [1, 1, 1], [2, 0, 1], [2, 1, 0]]
//   max_rows = 2
//   max_cols = 2 ->
//   [cyan, purple, cyan, cyan]
// Inputs:
//   partitions (array): Array of the AIE partitions [[partition #, row #, col
//   #],[...]] max_rows (integer): Number of rows in AIE grid max_cols
//   (integer): Number of columns in AIE grid
// Return:
//   colored_grid (array<array<string>>): Array of strings of colors for AIE
//   tiles, also containing the strings of the herd name
//   (optional): [[<color>, <name>], ...]
function color_grid(partitions: any, max_rows: number, max_cols: number) {

  var colored_grid: Array<Array<string>> = [];
  for (var x = 0; x < max_cols * max_rows; x++) {
    colored_grid[x] = [];
    colored_grid[x][0] = "white";
    colored_grid[x][1] = "";
  }
  for (var i = 0; i < partitions.length; i++) {
    const curr_row = partitions[i][1];
    const curr_col = partitions[i][2];
    const curr_color = select_color(partitions[i][0]);

    // This is a bit complex - the numbering for MLIR is a bit odd. The bottom
    // left tile is [1, 0] in the form <row, col>, as the shim is on the 0th
    // row. rows go up from there. Since the colors will be stored in a 1D grid,
    // the indexies need to be adjusted so that they end up in the right place.
    const index = (max_rows - (curr_row + 1)) * max_cols + curr_col;
    colored_grid[index][0] = String(curr_color);
    if (partitions[i].length === 4) {
      const name = adjust_name_size(partitions[i][3])
      colored_grid[index][1] = name;
    }
  }
  return colored_grid;
}

// Returns the first 4 characters of the name, as well as the number of the herd
// Example: asdfghj1234 -> asdf1234
function adjust_name_size(name: string) {
  const new_name_letters = name.replace(/[0-9]/g, '')
  if (new_name_letters.length <= 4) {
    return name;
  }
  else {
    return name.substring(0, 4) + name.replace(/^\D+/g, '')
  }
}

// Changes the format of the arrays to allow for easier coloring
// Example: [ [0, [1, 0], [2, 0], [3, 0]], [1, [1, 1], [2, 1]] ] ->
//          [ [0, 1, 0], [0, 2, 0], [0, 3, 0], [1, 1, 1], [1, 2, 1] ]
// Inputs:
//   partitions (Array): array of partitions: [ [Part #, [row, col] ...] ...]
// Outputs:
//   organized_array (Array<Array<int>>): organized array [ [part #, row, col]
//   ... ]
function rearrange_partition(partitions: any) {
  var organized_array = [];
  for (var i = 0; i < partitions.length; i++) {
    const current_part = partitions[i][0];
    var j = 1
    var has_name = false;
    if (typeof (partitions[i][1]) === 'string') {
      j = 2
      has_name = true;
    }
    for (j; j < partitions[i].length; j++) {
      var temp_array = [ current_part ];
      temp_array.push(partitions[i][j][0]);
      temp_array.push(partitions[i][j][1]);
      if (has_name) {
        temp_array.push(partitions[i][1]);
      }
      organized_array.push(temp_array);
    }
  }
  return organized_array;
}

// Selects the background color based on a partition number
// Example:
//   partition: 1 -> 'purple'
function select_color(partition: number) {
  if (partition > 127) {
    partition = partition % 127
  }
  if (partition === -1) {
    return 'white'
  } else {
    return CSS_COLOR_NAMES[partition]
  }
}

// creates the page of the partitions
function getWebviewContent(colored_grid: Array<Array<string>>|undefined,
                           region: Array<number>, routes: any,
                           route_number: number) {
  if (route_number > -1) {
    var colored_grid: Array<Array<string>>|undefined = [];
    for (var x = 0; x < row_size * col_size; x++) {
      colored_grid[x] = [];
      colored_grid[x][0] = "white";
      colored_grid[x][1] = "";
    }
    routes = JSON.stringify(routes);
  }
  const colored_grid_string = JSON.stringify(colored_grid);
  const color_options = JSON.stringify(CSS_COLOR_NAMES);

  return `<!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Partition View</title>
      <style>
        :root {
          --grid-cols: 1;
          --grid-rows: 1;
        }
        /* Change these values to change the size + spacing of the AIE div tiles */
        #container {
          display: inline-grid;
          grid-gap: 1px;
          grid-template-rows: repeat(var(--grid-rows), 50px);
          grid-template-columns: repeat(var(--grid-cols), 50px);
        }
        .grid-item {
		  display: flex;
		  justify-content: center;
		  align-items: center;
          padding: 1em;
          border: 1px solid black;
          color: black;
        }
      </style>
    </head>
      <body>
        <div id="container">
        </div>
        <script>
        const container = document.getElementById("container");
        
        const colored_grid = ${colored_grid_string};
        function makeRows(rows, cols) {

          // Setting the appropraite number of rows + columns
          container.style.setProperty('--grid-rows', rows);
          container.style.setProperty('--grid-cols', cols);

		  // Sets the style of the AIE elements (which color + location)
		  // Note cell 0 is the top left tile that is displayed, and cell 1
		  // is the tile to the right.
          for (c = 0; c < (rows * cols); c++) {
            let cell = document.createElement("div");
            cell.style.backgroundColor = colored_grid[c][0];
            const cell_row_num = ${row_size} - Math.floor(c / ${col_size}) - 1;
            const cell_col_num = c % ${col_size};
            const cell_string = String(cell_row_num) + "," + String(cell_col_num);
			if (colored_grid[c][1].length > 0) {
				cell.innerText = colored_grid[c][1];
			}
			cell.setAttribute('id', cell_string);
            
            container.appendChild(cell).className = "grid-item";

			let inner_text = document.createElement("div");
			const offsets = document.getElementById(cell_string).getBoundingClientRect();

			inner_text.style.position = 'absolute';
			inner_text.style.color = 'black';
			// These "magic number" control where coordinate text shows up in a tile.
			// for some reason. style.bottom gives horrid results. Spare yourself and avoid
			// this pain.
			inner_text.style.top = offsets.top + 33 + "px";
			inner_text.style.left = offsets.left + 29 + "px";
			inner_text.style.fontSize = "11px";
			inner_text.innerText = cell_string;
			
			document.body.appendChild(inner_text);
          };
        };
		// Draws regions around the AIE tiles
		// Inputs:
		//   row_size (int): The number of rows of AIE tiles that will be in each region
		//   col_size (int): The number of columns of AIE tiles that will be in each region
		//   grid_size_rows (int): The total number of rows in the AIE grid
		//   grid_size_col (int): The total number of columns in the AIE grid
		function makeRegions(row_size, col_size, grid_size_rows, grid_size_col) {
			for (var i = 0; i < grid_size_rows; i += row_size) {
				for (var j = 0; j < grid_size_col; j += col_size) {
					const top_right_string = String(i + row_size - 1) + "," + String(j + col_size - 1);
					const top_right_rect = document.getElementById(top_right_string).getBoundingClientRect();
					const bottom_left_string = String(i) + "," + String(j);
					const bottom_left_rect = document.getElementById(bottom_left_string).getBoundingClientRect();
					
					// The 4 is an offset because we use 2 pixel spacing between each AIE tile.
					const width = top_right_rect.right - bottom_left_rect.left - 4;
					const height = bottom_left_rect.bottom - top_right_rect.top - 4; 

					const htmlLine = "<div style=' \
											margin:0px; \
											padding: 0px; \
											border: 3px solid rgba(220, 220, 220, 1); \
											height:" + height + "px; \
											position:absolute; \
											left:" + bottom_left_rect.left + "px; \
											bottom:" + bottom_left_rect.bottom + "px; \
											top:" + top_right_rect.top + "px; \
											width:" + width + "px; ' />";
					document.body.innerHTML += htmlLine;
				}
			}
		};
        makeRows(${row_size}, ${col_size});
		if (${region.length} > 0 && ${route_number} < 0) {
			makeRegions(${region[0]}, ${region[1]}, ${row_size}, ${
      col_size});
		}
		if (${route_number} > -1) {
			const routes = ${routes};
			// gets the absolute values of the location of the divs
			const getOffset = (el) => {
				const rect = el.getBoundingClientRect();
				return {
				left: rect.left + window.pageXOffset,
				top: rect.top + window.pageYOffset,
				width: rect.width || el.offsetWidth,
				height: rect.height || el.offsetHeight
				};
			}
			
			const connectX = (div1, div2, color, thickness, offset) => {
				const off1 = getOffset(div1);
				const off2 = getOffset(div2);
			
				const x1 = off1.left + off1.width / 2;
				const y1 = off1.top + off1.height / 2;
			
				const x2 = off2.left + off2.width / 2;
				const y2 = off2.top + off2.height / 2;
			
				const length = Math.abs(x2 - x1);
			
				// controls absolute x + y positions
				const cx = ((x1 + x2) / 2) - (length / 2);
				const cy = ((y1 + y2) / 2) - (thickness / 2);
				left_loc = cx + offset;
				top_loc = cy + offset;
			
				const htmlLine = "<div style='padding:0px; \
											margin:0px; \
											height:" + thickness + "px; \
											background-color:" + color + "; \
											line-height:1px; \
											position:absolute; \
											left:" + left_loc + "px; \
											top:" + top_loc + "px; \
											width:" + length + "px; ' />";
				document.body.innerHTML += htmlLine;
			}
			const connectY = (div1, div2, color, thickness, offset) => {
				const off1 = getOffset(div1);
				const off2 = getOffset(div2);
			
				const x1 = off1.left + off1.width / 2;
				const y1 = off1.top + off1.height / 2;
			
				const x2 = off2.left + off2.width / 2;
				const y2 = off2.top + off2.height / 2;
			
				const length = Math.abs(y2 - y1);

				// controls absolute x + y positions
				const cx = 	((x1 + x2) / 2) - Math.abs((x2 - x1) / 2);
				const cy = ((y1 + y2) / 2) - Math.abs((y2 - y1) / 2);
				left_loc = cx + offset;
				top_loc = cy + offset;

				const htmlLine = "<div style='padding:0px; \
											margin:0px; \
											height:" + length + "px; \
											background-color:" + color + "; \
											line-height:1px; \
											position:absolute; \
											left:" + left_loc + "px; \
											top:" + top_loc + "px; \
											width:" + thickness + "px; ' />";
				document.body.innerHTML += htmlLine;
			}
			// Gets the adjacent tile based on the given direction
			// Returns:
			//   (array): [row, col]
			const get_adjacent_tile = (row, col, direction) => {
				if (direction === 'North') {
					return [row + 1, col];
				} else if (direction === 'South'){
					return [row - 1, col];
				} else if (direction === 'East') {
					return [row, col + 1];
				} else if (direction === 'West') {
					return [row, col - 1];
				}
			}

			// Drawing the wires
			const line_thickness = 2;
			var offset = 0;
			var color_options = ${color_options};
			for (var i = 0; i < ${route_number}; i++) {
				var line_color = color_options[i];
				for (var j = 0; j < routes[i].length; j++) {
					if (routes[i][j][0] === undefined) {
						break;
					}
					const row_source = routes[i][j][0][1];
					const col_source = routes[i][j][0][0];
					const source_id = String(row_source) + "," + String(col_source);
					
					if (j === 0) {

						// Label the first element as the source
						const rect = document.getElementById(source_id).getBoundingClientRect();
						let cell = document.createElement("div");
						cell.innerText = "Src";
						cell.style.position = "absolute";
						cell.style.textAlign = "center";
						cell.style.fontSize = "small";
						cell.style.top = String(rect.top + 32) + "px";
						cell.style.left = String(rect.left + 2) + "px";
						document.getElementById(source_id).appendChild(cell);  
						let circle = document.createElement("div");
						circle.style.position = "absolute";
						circle.style.width = "7px";
						circle.style.height = "7px";
						circle.style.backgroundColor = color_options[i];
						circle.style.top = String(rect.top + rect.height / 2 + offset) + "px";
						circle.style.left = String(rect.left + rect.width / 2 + offset - 2) + "px";
						// circle.style.borderRadius = "50%";
						document.getElementById(source_id).appendChild(circle); 

						// Draw a vertical line if the first element is on the bottom row 
						// (likely access via the shim)
						if (row_source === 0) {
						const rect = document.getElementById(source_id).getBoundingClientRect();
						const htmlLine = "<div style='padding:0px; \
											margin:0px; \
											height:" + String(rect.height / 2 - offset) + "px; \
											background-color:" + line_color + "; \
											position:absolute; \
											left:" + String(rect.left + rect.width / 2 + offset) + "px; \
											top:" + String(rect.top + rect.height / 2 + offset) + "px; \
											width:" + line_thickness + "px; ' />";
						document.body.innerHTML += htmlLine;
						}
					}
					for (var k = 0; k < routes[i][j][1].length; k++) {
						const direction = routes[i][j][1][k];
						// Adds the "DMA" text above the tile location
						if (direction === 'DMA' || direction === 'Core') {
							const rect = document.getElementById(source_id).getBoundingClientRect();
							let cell = document.createElement("div");
							cell.innerText = "DMA";
							cell.style.position = "absolute";
							cell.style.fontSize = "small";
							cell.style.top = String(rect.top) + "px";
							// How far to the right the DMA appears in the tile
							cell.style.left = String(rect.left + 2) + "px";
							document.getElementById(source_id).appendChild(cell);
							let circle = document.createElement("div");
							circle.style.position = "absolute";
							circle.style.width = "7px";
							circle.style.height = "7px";
							circle.style.backgroundColor = color_options[i];
							circle.style.top = String(rect.top + rect.height / 2 + offset - 2) + "px";
							circle.style.left = String(rect.left + rect.width / 2 + offset - 2) + "px";
							circle.style.borderRadius = "50%";
							document.getElementById(source_id).appendChild(circle); 
							continue;
						}
						if (row_source === 0 && direction === 'South') {
							const rect = document.getElementById(source_id).getBoundingClientRect();
							const htmlLine = "<div style='padding:0px; \
											margin:0px; \
											height:" + String(rect.height / 2 - offset) + "px; \
											background-color:" + line_color + "; \
											position:absolute; \
											left:" + String(rect.left + rect.width / 2 + offset) + "px; \
											top:" + String(rect.top + rect.height / 2 + offset) + "px; \
											width:" + line_thickness + "px; ' />";
							document.body.innerHTML += htmlLine;
							let cell = document.createElement("div");
							cell.innerText = "DMA";
							cell.style.position = "absolute";
							cell.style.textAlign = "center";
							cell.style.fontSize = "small";
							cell.style.top = String(rect.top) + "px";
							cell.style.left = String(rect.left + 2) + "px";
							document.getElementById(source_id).appendChild(cell); 
							document.getElementById(source_id).appendChild(cell);
							let circle = document.createElement("div");
							circle.style.position = "absolute";
							circle.style.width = "7px";
							circle.style.height = "7px";
							circle.style.backgroundColor = color_options[i];
							circle.style.top = String(rect.top + rect.height / 2 + offset - 2) + "px";
							circle.style.left = String(rect.left + rect.width / 2 + offset - 2) + "px";
							circle.style.borderRadius = "50%";
							document.getElementById(source_id).appendChild(circle);  
							continue;
						}
						const dest_XY = get_adjacent_tile(row_source, col_source, direction);
						const dest_xy_string = String(dest_XY[0]) + "," + String(dest_XY[1]);
						if (direction === 'North' || direction === 'South') {
							connectY(document.getElementById(source_id), document.getElementById(dest_XY), line_color, line_thickness, offset);
						}
						if (direction === 'East' || direction === 'West') {
							connectX(document.getElementById(source_id), document.getElementById(dest_XY), line_color, line_thickness, offset);
						}
					}
				}
				offset *= -1;
				if (i % 2 === 1) {
					offset = Math.abs(offset) + 2;
				}
				if (Math.abs(offset) >= 23) {
					offset = 0;
				}
			}
	    }
      </script>
    </body>
  </html>`;
}
