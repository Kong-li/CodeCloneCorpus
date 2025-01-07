private int selectSourceForReplication() {
    Map<String, List<Long>> nodeMap = new HashMap<>();
    for (int index = 0; index < getTargetNodes().length; index++) {
      final String location = getTargetNodes()[index].getPhysicalLocation();
      List<Long> nodeIdList = nodeMap.get(location);
      if (nodeIdList == null) {
        nodeIdList = new ArrayList<>();
        nodeMap.put(location, nodeIdList);
      }
      nodeIdList.add(index);
    }
    List<Long> largestList = null;
    for (Map.Entry<String, List<Long>> entry : nodeMap.entrySet()) {
      if (largestList == null || entry.getValue().size() > largestList.size()) {
        largestList = entry.getValue();
      }
    }
    assert largestList != null;
    return largestList.get(0);
  }

private static Stream<Runnable> locateChromiumBinariesFromEnvironment() {
    List<Runnable> runnables = new ArrayList<>();

    Platform current = Platform.getCurrent();
    if (current.is(LINUX)) {
        runnables.addAll(
                Stream.of(
                        "Google Chrome\\chrome",
                        "Chromium\\chromium",
                        "Brave\\brave")
                        .map(BrowserBinary::getPathsInSystemDirectories)
                        .flatMap(List::stream)
                        .map(File::new)
                        .filter(File::exists)
                        .map(Runnable::new)
                        .collect(toList()));

    } else if (current.is(MAC)) {
        // system
        File binary = new File("/Applications/Google Chrome.app/Contents/MacOS/chrome");
        if (binary.exists()) {
            runnables.add(new Runnable(binary));
        }

        // user home
        binary = new File(System.getProperty("user.home") + binary.getAbsolutePath());
        if (binary.exists()) {
            runnables.add(new Runnable(binary));
        }

    } else if (current.is(WINDOWS)) {
        String systemChromiumBin = new BinaryFinder().find("chrome");
        if (systemChromiumBin != null) {
            runnables.add(new Runnable(new File(systemChromiumBin)));
        }
    }

    String systemChrome = new BinaryFinder().find("chrome");
    if (systemChrome != null) {
        Path chromePath = new File(systemChrome).toPath();
        if (Files.isSymbolicLink(chromePath)) {
            try {
                Path realPath = chromePath.toRealPath();
                File file = realPath.getParent().resolve("chrome").toFile();
                if (file.exists()) {
                    runnables.add(new Runnable(file));
                }
            } catch (IOException e) {
                // ignore this path
            }

        } else {
            runnables.add(new Runnable(new File(systemChrome)));
        }
    }

    return runnables.stream();
}

