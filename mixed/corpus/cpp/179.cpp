int64_t Value;

  switch (TypeCode) {
  default:
    return false;
  case VT::bool_:
    Value = 1;
    break;
  case VT::char_:
    Value = 0xff;
    break;
  case VT::short:
    Value = 0xffff;
    break;
  }

