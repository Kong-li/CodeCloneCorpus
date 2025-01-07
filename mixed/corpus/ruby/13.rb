      def write_object(object, packer)
        if object.class.respond_to?(:from_msgpack_ext)
          packer.write(LOAD_WITH_MSGPACK_EXT)
          write_class(object.class, packer)
          packer.write(object.to_msgpack_ext)
        elsif object.class.respond_to?(:json_create)
          packer.write(LOAD_WITH_JSON_CREATE)
          write_class(object.class, packer)
          packer.write(object.as_json)
        else
          raise_unserializable(object)
        end

