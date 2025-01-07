export async function addNewUser({ userLogin, passWord }) {
  // Here you should create the user and save the salt and hashed password (some dbs may have
  // authentication methods that will do it for you so you don't have to worry about it):
  const saltKey = crypto.randomBytes(16).toString("hex");
  const hashValue = crypto
    .pbkdf2Sync(passWord, saltKey, 1000, 64, "sha512")
    .toString("hex");
  const newUser = {
    id: uuidv4(),
    creationTime: Date.now(),
    userLogin,
    hashValue,
    saltKey,
  };

  // This is an in memory store for users, there is no data persistence without a proper DB
  users.push(newUser);

  return { userLogin, creationTime: Date.now() };
}

