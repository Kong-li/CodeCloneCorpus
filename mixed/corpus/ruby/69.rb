      def parameter_filtered_location
        uri = URI.parse(location)
        unless uri.query.nil? || uri.query.empty?
          parts = uri.query.split(/([&;])/)
          filtered_parts = parts.map do |part|
            if part.include?("=")
              key, value = part.split("=", 2)
              request.parameter_filter.filter(key => value).first.join("=")
            else
              part
            end

