def initialize(self, ec_type, etype):
    data_arr = np.array(
        [str(i) * 15 for i in range(200_000)], dtype=self.et_mapping[etype]
    )
    if ec_type == "data":
        self.data = data_arr
    elif ec_type == "table":
        self.data = data_arr.reshape((100_000, 3)).copy()
    elif ec_type == "category_data":
        # GH45678. Testing construction of string tables from ExtensionArrays
        self.data = Categorical(data_arr)

def _conv1d_custom_operation(
    tensor_in: torch.Tensor,
    kernel: torch.Tensor,
    offset: Optional[torch.Tensor],
    stride_params: List[int],
    padding_params: List[int],
    dilation_factors: List[int],
    group_count: int,
) -> torch.Tensor:
    # The original conv1d operation is adapted to a 2D convolution by adding and removing dimensions.
    tensor_in_expanded = torch.unsqueeze(tensor_in, dim=2)
    conv_result_2d = torch.nn.functional.conv2d(
        input=tensor_in_expanded,
        weight=kernel,
        bias=offset,
        stride=stride_params,
        padding=padding_params,
        dilation=dilation_factors,
        groups=group_count
    )
    conv_output_resized = torch.squeeze(conv_result_2d, dim=2)
    return conv_output_resized

def _fetch_fer2013_faces(
    data_folder_path, slice_=None, grayscale=False, resize=None, min_faces_per_person=0
):
    """Perform the actual data loading for the fer2013 faces dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace("_", " ")
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError(
            "min_faces_per_person=%d is too restrictive" % min_faces_per_person
        )

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = _load_imgs(file_paths, slice_, grayscale, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    return faces, target, target_names

def validate_editor_link(self, person_id):
        """
        FK reverse relations are represented by managers and can be manipulated similarly.
        """
        other_db_person = Person.objects.using("other").get(pk=person_id)
        book_obj = Book.objects.using("other").filter(editor=other_db_person, pk=1).first()
        if book_obj:
            editor_name = book_obj.editor.name
            default_manager = other_db_person.edited.db_manager(using="default")
            db_check = default_manager.db
            all_books = default_manager.all()
            self.assertEqual(db_check, "default")
            self.assertEqual(all_books.db, "default")

