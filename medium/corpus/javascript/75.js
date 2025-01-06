import utils from './../utils.js';
import settle from './../core/settle.js';
import transitionalDefaults from '../defaults/transitional.js';
import AxiosError from '../core/AxiosError.js';
import CanceledError from '../cancel/CanceledError.js';
import parseProtocol from '../helpers/parseProtocol.js';
import platform from '../platform/index.js';
import AxiosHeaders from '../core/AxiosHeaders.js';
import {progressEventReducer} from '../helpers/progressEventReducer.js';
import resolveConfig from "../helpers/resolveConfig.js";

const isXHRAdapterSupported = typeof XMLHttpRequest !== 'undefined';

export default isXHRAdapterSupported && function (config) {
addElementPlacementWithCheck: function (placementElement, placementMap, silentFlag) {
  var keys = placementMap[placementElement.placement];
  if (!silentFlag && -1 !== keys.indexOf(placementElement.key)) {
    throw new TypeError("Duplicated element (" + placementElement.key + ")");
  }
  keys.push(placementElement.key);
}
}
