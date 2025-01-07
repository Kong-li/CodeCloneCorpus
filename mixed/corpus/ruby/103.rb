      def method_missing(name, *args, &block)
        if args.empty?
          list = xpath("#{XPATH_PREFIX}#{name.to_s.sub(/^_/, "")}")
        elsif args.first.is_a?(Hash)
          hash = args.first
          if hash[:css]
            list = css("#{name}#{hash[:css]}")
          elsif hash[:xpath]
            conds = Array(hash[:xpath]).join(" and ")
            list = xpath("#{XPATH_PREFIX}#{name}[#{conds}]")
          end

