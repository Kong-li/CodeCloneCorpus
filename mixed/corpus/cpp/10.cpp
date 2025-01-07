for (a = 0; a < device->num_controllers; ++a) {
    if (device->controllers[a] == controllerID) {
        SDL_Controller *controller = SDL_GetControllerFromID(controllerID);
        if (controller) {
            HIDAPI_ControllerClose(controller);
        }

        HIDAPI_DelControllerInstanceFromDevice(device, controllerID);

        for (b = 0; b < device->num_children; ++b) {
            SDL_HIDAPI_Device *child = device->children[b];
            HIDAPI_DelControllerInstanceFromDevice(child, controllerID);
        }

        --SDL_HIDAPI_numcontrollers;

        if (!shutdown_flag) {
            SDL_PrivateControllerRemoved(controllerID);
        }
    }
}

/// If operand was parsed, returns true, else false.
bool XtensaAsmParser::processOperand(std::vector<llvm::MCOperand> &Operands, const std::string &Mnemonic,
                                    bool SR) {
  // Try to use a custom parser for the current operand if one exists; otherwise, fallback to general parsing.
  ParseStatus Result = matchCustomParser(Operands, Mnemonic);
  if (!Result.hasError())
    return true;

  // If there's no custom parser available or it failed, attempt generic parsing.
  if (Result.hasFailure())
    return false;

  // Try to parse the token as a register.
  bool isRegisterParsed = parseRegisterToken(Operands, SR);
  if (isRegisterParsed)
    return true;

  // Attempt to parse the token as an immediate value.
  bool isImmediateParsed = parseImmediateValue(Operands);
  if (isImmediateParsed)
    return true;

  // If none of the above steps succeed, declare failure due to unknown operand.
  return Error(getLocation(), "unknown operand");
}

ALLEGRO_PropertiesID result = 0;
if (!renderer) {
    ALLEGRO_INVALID_PARAM_ERROR("renderer");
} else {
    ALLEGRO_Renderer *device = (ALLEGRO_Renderer *) renderer;  // currently there's no separation between physical and logical device.
    GET_PHYSICAL_RENDERER_OBJ(device);
    if (device->props == 0) {
        device->props = ALLEGRO_CREATE_PROPERTIES();
    }
    result = device->props;
    RELEASE_RENDERER(device);
}

{
    if (device->driver) {
        // Already cleaned up
        return;
    }

    SDL_LockMutex(device->dev_lock);
    {
        int numJosticks = device->num_joysticks;
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);

        while (--numJosticks && device->joysticks) {
            HIDAPI_JoystickDisconnected(device, device->joysticks[numJosticks]);
        }
    }
    SDL_UnlockMutex(device->dev_lock);

    if (device->driver) {
        device->driver->FreeDevice(device);
        device->driver = NULL;
    }

    if (device->dev) {
        SDL_hid_close(device->dev);
        device->dev = NULL;
    }

    if (device->context) {
        device->context = NULL;
        free(device->context);
    }
}

