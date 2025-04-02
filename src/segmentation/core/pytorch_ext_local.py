import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import torch


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-8):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        
        # dice is undefined when target is all zeros
        if target.sum() == 0:
            return torch.tensor([0], device=predicted.device)
    
        # Flatten the predictions and targets
        predicted_flat = predicted.flatten()
        target_flat = target.flatten()

        # Intersection and Union
        intersection = torch.sum(predicted_flat * target_flat)
        union = torch.sum(predicted_flat) + torch.sum(target_flat)

        # Dice Coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss
        dice_loss = 1.0 - dice_coefficient

        return dice_loss



class F1MeasureMultich:
    """
    A class for computing the F1 score for multiclass segmentation tasks,
    along with defect counts per image.

    Attributes:
        class_stats (dict): Stores tp, fp, fn counts for each class.
        class_counts (dict): Stores the number of images containing at least one pixel for each class.

    Methods:
        update(preds: torch.Tensor, gds: torch.Tensor) -> dict:
            Updates class stats and defect counts, then returns F1 scores per class.

        get() -> dict:
            Computes and returns the F1 scores for each class.

        get_counts() -> dict:
            Returns the number of images containing at least one pixel of each class.
    """

    def __init__(self, classes_names) -> None:
        """
        Initializes the F1Measure object with dictionaries to store class statistics
        and defect counts per class.
        """
        self.class_stats = {}
        self.class_counts = {}
        self.classes_names = classes_names

    def update(self, preds: torch.Tensor, gds: torch.Tensor) -> dict:
        """
        Updates the counts for each class based on predictions and ground truth.

        Args:
            preds (torch.Tensor): Tensor of predicted values (shape: [batch, num_classes, H, W]).
            gds (torch.Tensor): One-hot encoded ground truth tensor (shape: [batch, num_classes, H, W]).

        Returns:
            dict: The updated F1 scores per class.
        """
        num_classes = preds.shape[1]  # Number of classes

        for c in range(num_classes):
            y_hat = preds[:, c].to(torch.bool)  # Convert predictions to boolean
            y = gds[:, c].to(torch.bool)  # Convert ground truth to boolean

            tp = torch.logical_and(y_hat, y).sum().item()  # True Positives
            fp = torch.logical_and(y_hat, ~y).sum().item()  # False Positives
            fn = torch.logical_and(~y_hat, y).sum().item()  # False Negatives

            # Initialize class stats if not present
            if c not in self.class_stats:
                self.class_stats[c] = {"tp": 0, "fp": 0, "fn": 0}

            self.class_stats[c]["tp"] += tp
            self.class_stats[c]["fp"] += fp
            self.class_stats[c]["fn"] += fn

            # Count images that contain at least one pixel of this class
            class_present = (y.sum(dim=(1, 2)) > 0).sum().item()  # Count images where class c appears
            self.class_counts[c] = self.class_counts.get(c, 0) + class_present

        return self.get()  # Return updated F1 scores

    def get(self) -> dict:
        """
        Computes and returns the F1 scores per class.

        Returns:
            dict: A dictionary mapping class names to their F1 scores.
        """
        f1_scores = {}
        for c, stats in self.class_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            f1_scores[c] = (2 * tp) / ((2 * tp + fp + fn) + 1e-6)

        # convert with class names
        return {self.classes_names[c]: f1 for c, f1 in f1_scores.items()}


    def get_counts(self) -> dict:
        """
        Returns the number of images that contain at least one pixel of each class.

        Returns:
            dict: A dictionary mapping class names to the number of images where they appear.
        """
        return {self.classes_names[c]: count for c, count in self.class_counts.items()}




class F1Measure:
    """
    A class for computing the F1 score for binary classification tasks.

    Attributes:
        tp (int): The number of true positives.
        fp (int): The number of false positives.
        fn (int): The number of false negatives.

    Methods:
        update(preds: torch.Tensor, gds: torch.Tensor) -> float:
            Updates the true positive, false positive, and false negative counts
            based on the predicted and ground truth values, and returns the current F1 score.

        get() -> float:
            Computes and returns the F1 score based on the current counts.
    """

    def __init__(self) -> None:
        """
        Initializes the F1Measure object with zeroed counts for 
        true positives, false positives, and false negatives.
        """
        self.tp: int = 0  # True positives count
        self.fp: int = 0  # False positives count
        self.fn: int = 0  # False negatives count

    def update(self, preds: torch.Tensor, gds: torch.Tensor) -> float:
        """
        Updates the true positive, false positive, and false negative counts 
        based on the given predictions and ground truth values.

        Args:
            preds (torch.Tensor): A tensor of predicted values (binary format).
            gds (torch.Tensor): A tensor of ground truth values (binary format).

        Returns:
            float: The updated F1 score after processing the batch.
        """

        for y_hat, y in zip(preds, gds):
            y_hat = y_hat.to(torch.bool)  # Convert predictions to boolean
            y = y.to(torch.bool)  # Convert ground truth to boolean

            self.tp += torch.logical_and(y_hat, y).sum().item()  # Count true positives
            self.fp += torch.logical_and(y_hat, ~y).sum().item()  # Count false positives
            self.fn += torch.logical_and(~y_hat, y).sum().item()  # Count false negatives

        return self.get()  # Return updated F1 score

    def get(self) -> float:
        """
        Computes and returns the F1 score based on the current counts.

        Returns:
            float: The computed F1 score.
        """
        return (2 * self.tp) / ((2 * self.tp + self.fp + self.fn) + 1e-6)



class ConfusionMatrix:
    """
    A class for computing the Confusion Matrix for binary classification tasks.

    Attributes:
        tp (int): The number of true positives.
        fp (int): The number of false positives.
        fn (int): The number of false negatives.
        tn (int): The number of true negatives.

    Methods:
        update(preds: torch.Tensor, gds: torch.Tensor) -> float:
            Updates the values of the confusion matrix based on the predicted and ground truth values.

        get() -> tuple:
            Returns the values of the confusion matrix.
    """

    def __init__(self) -> None:
        self.tp: int = 0  # True positives count
        self.fp: int = 0  # False positives count
        self.fn: int = 0  # False negatives count
        self.tn: int = 0  # True negatives count

    def update(self, preds: torch.Tensor, gds: torch.Tensor) -> tuple:
        """
        Args:
            preds (torch.Tensor): A tensor of predicted values (binary format).
            gds (torch.Tensor): A tensor of ground truth values (binary format).

        Returns:
            tuple: The updated values of the confusion matrix.
        """

        for y_hat, y in zip(preds, gds):
            y_hat = y_hat.to(torch.bool)  # Convert predictions to boolean
            y = y.to(torch.bool)  # Convert ground truth to boolean

            self.tp += torch.logical_and(y_hat, y).sum().item()  # Count true positives
            self.fp += torch.logical_and(y_hat, ~y).sum().item()  # Count false positives
            self.fn += torch.logical_and(~y_hat, y).sum().item()  # Count false negatives
            self.tn += torch.logical_and(~y_hat, ~y).sum().item() # Count true negatives

        return self.get()  # Return updated F1 score


    def get(self) -> tuple:
        """
        Returns the values of the confusion matrix.

        Returns:
            tuple: The values of the confusion matrix.
        """
        return self.tp, self.fp, self.fn, self.tn
    

    def pretty_print(self):
        print(f"TP: {self.tp}, FP: {self.fp}, FN: {self.fn}, TN: {self.tn}")


    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-6)


    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-6)

