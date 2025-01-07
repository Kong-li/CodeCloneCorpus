          def serialize(value)
            case value
            when ActiveRecord::Point
              "(#{number_for_point(value.x)},#{number_for_point(value.y)})"
            when ::Array
              serialize(build_point(*value))
            when ::Hash
              serialize(build_point(*values_array_from_hash(value)))
            else
              super
            end

    def send_blob_byte_range_data(blob, range_header, disposition: nil)
      ranges = Rack::Utils.get_byte_ranges(range_header, blob.byte_size)

      return head(:range_not_satisfiable) if ranges.blank? || ranges.all?(&:blank?)

      if ranges.length == 1
        range = ranges.first
        content_type = blob.content_type_for_serving
        data = blob.download_chunk(range)

        response.headers["Content-Range"] = "bytes #{range.begin}-#{range.end}/#{blob.byte_size}"
      else
        boundary = SecureRandom.hex
        content_type = "multipart/byteranges; boundary=#{boundary}"
        data = +""

        ranges.compact.each do |range|
          chunk = blob.download_chunk(range)

          data << "\r\n--#{boundary}\r\n"
          data << "Content-Type: #{blob.content_type_for_serving}\r\n"
          data << "Content-Range: bytes #{range.begin}-#{range.end}/#{blob.byte_size}\r\n\r\n"
          data << chunk
        end

