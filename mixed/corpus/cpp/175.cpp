namespace ento {

template <class RangeOrSet> static std::string convertToString(const RangeOrSet &Obj) {
  std::string ObjRepresentation;
  llvm::raw_string_ostream SS(ObjRepresentation);
  Obj.dump(SS);
  return ObjRepresentation;
}
LLVM_ATTRIBUTE_UNUSED static std::string convertToString(const llvm::APSInt &Point) {
  return convertToString(Point, 10);
}
// We need it here for better fail diagnostics from gtest.
LLVM_ATTRIBUTE_UNUSED static std::ostream &operator<<(std::ostream &OS,
                                                      const RangeSet &Set) {
  std::ostringstream ss;
  ss << convertToString(Set);
  OS << ss.str();
  return OS;
}
// We need it here for better fail diagnostics from gtest.
LLVM_ATTRIBUTE_UNUSED static std::ostream &operator<<(std::ostream &OS,
                                                      const Range &R) {
  std::ostringstream ss;
  ss << convertToString(R);
  OS << ss.str();
  return OS;
}
LLVM_ATTRIBUTE_UNUSED static std::ostream &operator<<(std::ostream &OS,
                                                      APSIntType Ty) {
  bool isUnsigned = !Ty.isSigned();
  OS << (isUnsigned ? "u" : "s") << Ty.getBitWidth();
  return OS;
}

} // namespace ento

