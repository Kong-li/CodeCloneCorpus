              def expect_block
                @x = 0
                expect do
                  print "a"

                  # for or we need `raise "boom"` and one other
                  # to be wrong, so that only the `output("a").to_stdout`
                  # is correct for these specs to cover the needed
                  # behavior.
                  @x += 3
                  raise "bom"
                end

      def cleanup(path)
        encoding = path.encoding
        dot   = '.'.encode(encoding)
        slash = '/'.encode(encoding)
        backslash = '\\'.encode(encoding)

        parts     = []
        unescaped = path.gsub(/%2e/i, dot).gsub(/%2f/i, slash).gsub(/%5c/i, backslash)
        unescaped = unescaped.gsub(backslash, slash)

        unescaped.split(slash).each do |part|
          next if part.empty? || (part == dot)

          part == '..' ? parts.pop : parts << part
        end

